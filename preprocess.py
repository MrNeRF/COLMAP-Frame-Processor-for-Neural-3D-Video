import os
import glob
import shutil
import logging
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def do_system(cmd, error_msg="Command failed"):
    """Execute system command and handle errors."""
    logging.info(f"Running: {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"{error_msg} with code {exit_code}")
        raise RuntimeError(f"{error_msg} with code {exit_code}")
    return exit_code

def process_poses(path):
    """Process camera poses and group frames by timestamp."""
    images = sorted([str(p.relative_to(path)) for p in Path(path/"images").glob("*.[pj][pn][g]")])
    if not images:
        raise RuntimeError("No images found in images directory")
    
    poses_bounds = np.load(path / 'poses_bounds.npy')
    N = poses_bounds.shape[0]
    
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    bounds = poses_bounds[:, -2:]
    H, W, fl = poses[0, :, -1]
    
    poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses = np.concatenate([poses, last_row], axis=1)
    
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1
    
    frames_by_time = defaultdict(list)
    cams = sorted(set([im.split('_')[0] for im in images]))
    
    for i, cam in enumerate(cams):
        cam_images = [im for im in images if cam in im]
        for img in cam_images:
            time = int(img.rsplit('_', 1)[1].split('.')[0])
            frames_by_time[time].append({
                'file_path': img,
                'transform_matrix': poses[i].tolist(),
                'camera_id': i
            })
    
    return frames_by_time, {'w': W, 'h': H, 'fl_x': fl, 'fl_y': fl, 'cx': W // 2, 'cy': H // 2}

def run_colmap_for_frame(frame_time, frame_data, camera_info, base_path, use_gpu=True):
    """Run COLMAP for a specific timestamp."""
    frame_dir = base_path / f"frame_{frame_time:04d}"
    temp_dir = frame_dir / "temp"
    
    # Create temporary directory structure for COLMAP processing
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "sparse").mkdir(parents=True, exist_ok=True)
    
    # Create final directory structure
    images_dir = frame_dir / "images"
    sparse_dir = frame_dir / "sparse"
    sparse_0_dir = sparse_dir / "0"
    sparse_0_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to temporary input directory
    for frame in frame_data:
        src = base_path / frame['file_path']
        dst = temp_dir / src.name
        if not dst.exists():
            os.symlink(src.absolute(), dst)
    
    colmap_command = "colmap"
    
    # Run COLMAP pipeline
    do_system(
        f"{colmap_command} feature_extractor "
        f"--database_path {temp_dir}/database.db "
        f"--image_path {temp_dir} "
        f"--ImageReader.single_camera 0 "
        f"--ImageReader.camera_model PINHOLE "
        f"--SiftExtraction.use_gpu {int(use_gpu)}",
        "Feature extraction failed"
    )
    
    do_system(
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {temp_dir}/database.db "
        f"--SiftMatching.use_gpu {int(use_gpu)}",
        "Feature matching failed"
    )
    
    do_system(
        f"{colmap_command} mapper "
        f"--database_path {temp_dir}/database.db "
        f"--image_path {temp_dir} "
        f"--output_path {temp_dir}/sparse "
        f"--Mapper.ba_global_function_tolerance=0.000001",
        "Mapper failed"
    )
    
    # Run image_undistorter
    do_system(
        f"{colmap_command} image_undistorter "
        f"--image_path {temp_dir} "
        f"--input_path {temp_dir}/sparse/0 "
        f"--output_path {frame_dir} "
        f"--output_type COLMAP",
        "Image undistortion failed"
    )
    
    # Move all COLMAP output files to sparse/0
    for file in ['cameras.bin', 'images.bin', 'points3D.bin']:
        if (sparse_dir / file).exists():
            shutil.move(str(sparse_dir / file), str(sparse_0_dir / file))
    
    # Clean up temporary files
    shutil.rmtree(temp_dir)
    
    # Remove any extra directories created by COLMAP
    for item in frame_dir.iterdir():
        if item.name not in ['images', 'sparse']:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    logging.info(f"Completed processing frame_{frame_time:04d}")

def extract_frames(video_path, output_path):
    """Extract frames from videos using ffmpeg."""
    videos = [str(p) for p in Path(video_path).glob("*.mp4")]
    if not videos:
        logging.warning("No MP4 files found in the input directory")
        return
    
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    for video in videos:
        cam_name = Path(video).stem
        do_system(
            f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png",
            "Frame extraction failed"
        )
    
    logging.info(f"Extracted frames from {len(videos)} videos to {images_path}")

def main():
    parser = argparse.ArgumentParser(description="Process multi-camera data frame by frame")
    parser.add_argument("path", help="Input path to the data directory")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from videos")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage in COLMAP")
    args = parser.parse_args()
    
    base_path = Path(args.path).resolve()
    if not base_path.exists():
        raise RuntimeError(f"Input path {base_path} does not exist")
    
    # Extract frames if requested
    if args.extract_frames:
        extract_frames(base_path, base_path)
        logging.info("Frame extraction completed")
    
    # Process poses and get frames grouped by timestamp
    frames_by_time, camera_info = process_poses(base_path)
    
    # Process each timestamp
    for time, frame_data in sorted(frames_by_time.items()):
        logging.info(f"\nProcessing frame {time}")
        run_colmap_for_frame(time, frame_data, camera_info, base_path, not args.no_gpu)
