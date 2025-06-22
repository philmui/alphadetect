#!/usr/bin/env python3
"""
AlphaDetect CLI - Human Pose Detection Tool

This tool provides pose detection capabilities using multiple backends:
- MediaPipe (primary, most reliable)
- Ultralytics YOLO (alternative)
- AlphaPose (if available)

Author: AlphaDetect Team
Date: 2025-06-22
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# Check available backends
MEDIAPIPE_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
ALPHAPOSE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe backend available")
except ImportError:
    logger.warning("MediaPipe not available")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("Ultralytics YOLO backend available")
except ImportError:
    logger.warning("Ultralytics YOLO not available")

try:
    import torch
    from alphapose.models import builder
    from alphapose.utils.config import update_config
    ALPHAPOSE_AVAILABLE = True
    logger.info("AlphaPose backend available")
except ImportError as e:
    logger.warning(f"AlphaPose not available: {e}")


class AlphaDetectConfig:
    """Configuration for AlphaDetect CLI."""
    
    # Default paths
    DEFAULT_OUTPUT_DIR = Path("outputs")
    
    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from parsed arguments."""
        self.input_path: Path = Path(args.video) if args.video else Path(args.image_dir)
        self.is_video: bool = args.video is not None
        
        # Create output directory if it doesn't exist
        self.output_dir: Path = Path(args.output_dir) if args.output_dir else self.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate timestamp for output files
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set output JSON path
        if args.output:
            self.output_json: Path = Path(args.output)
        else:
            filename = f"pose_{self.timestamp}.json"
            self.output_json = self.output_dir / filename
        
        # Create frame and overlay directories
        input_name = self.input_path.stem
        self.frames_dir: Path = self.output_dir / f"frames_{input_name}_{self.timestamp}"
        self.overlay_dir: Path = self.output_dir / f"overlay_{input_name}_{self.timestamp}"
        
        self.frames_dir.mkdir(exist_ok=True)
        self.overlay_dir.mkdir(exist_ok=True)
        
        # Backend selection
        self.backend: str = args.backend
        self.debug: bool = args.debug
        self.min_confidence: float = args.min_confidence
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Configuration initialized: backend={self.backend}, input={self.input_path}")
    
    def _validate(self) -> None:
        """Validate the configuration."""
        # Check if input exists
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        # Check if output directory is writable
        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {self.output_dir}")


class MediaPipeDetector:
    """MediaPipe-based pose detector."""
    
    def __init__(self, config: AlphaDetectConfig):
        """Initialize MediaPipe pose detector."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not available")
        
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=config.min_confidence,
            min_tracking_confidence=config.min_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        logger.info("MediaPipe pose detector initialized")
    
    def detect_poses(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses in a single image."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        poses = []
        if results.pose_landmarks:
            # Convert landmarks to our format
            landmarks = results.pose_landmarks.landmark
            height, width = image.shape[:2]
            
            # Calculate bounding box
            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]
            
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            # Convert landmarks to keypoints
            keypoints = []
            for lm in landmarks:
                x, y = lm.x * width, lm.y * height
                confidence = lm.visibility if hasattr(lm, 'visibility') else 1.0
                keypoints.append([x, y, confidence])
            
            pose = {
                'frame_idx': frame_idx,
                'bbox': [x1, y1, x2, y2, 1.0],  # confidence = 1.0 for MediaPipe
                'score': 1.0,
                'keypoints': keypoints,
                'backend': 'mediapipe'
            }
            poses.append(pose)
        
        return poses
    
    def draw_poses(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw poses on the image."""
        if not poses:
            return image
        
        # Convert to RGB for MediaPipe drawing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for pose in poses:
            # Draw bounding box
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Convert keypoints back to MediaPipe format for drawing
            if len(pose['keypoints']) == 33:  # MediaPipe has 33 landmarks
                # Create a mock landmark list
                height, width = image.shape[:2]
                landmark_list = []
                
                for x, y, conf in pose['keypoints']:
                    # Create a simple object with x, y attributes
                    class MockLandmark:
                        def __init__(self, x, y):
                            self.x = x / width
                            self.y = y / height
                    
                    landmark_list.append(MockLandmark(x, y))
                
                # Draw pose connections
                self._draw_landmarks(image, landmark_list)
        
        return image
    
    def _draw_landmarks(self, image: np.ndarray, landmarks: List) -> None:
        """Draw pose landmarks and connections."""
        # MediaPipe pose connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        
        height, width = image.shape[:2]
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        
        # Draw connections
        for connection in connections:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                start_landmark = landmarks[connection[0]]
                end_landmark = landmarks[connection[1]]
                
                start_x, start_y = int(start_landmark.x * width), int(start_landmark.y * height)
                end_x, end_y = int(end_landmark.x * width), int(end_landmark.y * height)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)


class UltralyticsDetector:
    """Ultralytics YOLO-based pose detector."""
    
    def __init__(self, config: AlphaDetectConfig):
        """Initialize Ultralytics YOLO pose detector."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not available")
        
        self.config = config
        self.model = YOLO('yolov8n-pose.pt')  # Load YOLOv8 pose model
        
        logger.info("Ultralytics YOLO pose detector initialized")
    
    def detect_poses(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses in a single image."""
        # Run YOLO inference
        results = self.model(image, verbose=False)
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    kpts = keypoints[i]
                    
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_conf = box.conf[0].cpu().numpy()
                    
                    # Skip low-confidence detections
                    if box_conf < self.config.min_confidence:
                        continue
                    
                    # Extract keypoints
                    kpts_data = kpts.data[0].cpu().numpy()  # Shape: (17, 3) for COCO format
                    keypoints_list = []
                    
                    for j in range(len(kpts_data)):
                        x, y, conf = kpts_data[j]
                        keypoints_list.append([float(x), float(y), float(conf)])
                    
                    pose = {
                        'frame_idx': frame_idx,
                        'bbox': [float(x1), float(y1), float(x2), float(y2), float(box_conf)],
                        'score': float(box_conf),
                        'keypoints': keypoints_list,
                        'backend': 'ultralytics'
                    }
                    poses.append(pose)
        
        return poses
    
    def draw_poses(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw poses on the image."""
        # COCO keypoint pairs for skeleton
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170)
        ]
        
        for pose in poses:
            # Draw bounding box
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            keypoints = pose['keypoints']
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > self.config.min_confidence:
                    x, y = int(x), int(y)
                    color = colors[i % len(colors)]
                    cv2.circle(image, (x, y), 4, color, -1)
            
            # Draw skeleton
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints)):
                    x1, y1, conf1 = keypoints[pt1_idx]
                    x2, y2, conf2 = keypoints[pt2_idx]
                    
                    if conf1 > self.config.min_confidence and conf2 > self.config.min_confidence:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        color = colors[pt1_idx % len(colors)]
                        cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        return image


class PoseDetector:
    """Main pose detector that manages different backends."""
    
    def __init__(self, config: AlphaDetectConfig):
        """Initialize the pose detector with the best available backend."""
        self.config = config
        self.detector = None
        
        # Select backend
        if config.backend == "auto":
            # Auto-select the best available backend
            if MEDIAPIPE_AVAILABLE:
                self.backend_name = "mediapipe"
                self.detector = MediaPipeDetector(config)
            elif ULTRALYTICS_AVAILABLE:
                self.backend_name = "ultralytics"
                self.detector = UltralyticsDetector(config)
            else:
                raise RuntimeError("No pose detection backends available. Please install MediaPipe or Ultralytics.")
        elif config.backend == "mediapipe":
            if not MEDIAPIPE_AVAILABLE:
                raise RuntimeError("MediaPipe backend not available. Please install MediaPipe.")
            self.backend_name = "mediapipe"
            self.detector = MediaPipeDetector(config)
        elif config.backend == "ultralytics":
            if not ULTRALYTICS_AVAILABLE:
                raise RuntimeError("Ultralytics backend not available. Please install Ultralytics.")
            self.backend_name = "ultralytics"
            self.detector = UltralyticsDetector(config)
        elif config.backend == "alphapose":
            if not ALPHAPOSE_AVAILABLE:
                raise RuntimeError("AlphaPose backend not available. Please install AlphaPose correctly.")
            # TODO: Implement AlphaPose backend if needed
            raise NotImplementedError("AlphaPose backend not implemented in this version. Use MediaPipe or Ultralytics.")
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
        
        logger.info(f"Using backend: {self.backend_name}")
    
    def process_video(self) -> List[Dict[str, Any]]:
        """Process a video file and return pose data."""
        video_path = str(self.config.input_path)
        logger.info(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Process frames
        all_poses = []
        frame_idx = 0
        
        with logger.contextualize(video=video_path):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Log progress
                if frame_idx % 100 == 0 or frame_idx == 0:
                    progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
                    logger.info(f"Processing frame {frame_idx}/{total_frames} ({progress:.1f}%)")
                
                # Save raw frame
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(self.config.frames_dir / frame_filename), frame)
                
                # Detect poses
                poses = self.detector.detect_poses(frame, frame_idx)
                all_poses.extend(poses)
                
                # Draw and save overlay frame
                overlay_frame = self.detector.draw_poses(frame.copy(), poses)
                cv2.imwrite(str(self.config.overlay_dir / frame_filename), overlay_frame)
                
                frame_idx += 1
        
        cap.release()
        logger.info(f"Video processing complete: {frame_idx} frames processed, {len(all_poses)} poses detected")
        
        return all_poses
    
    def process_images(self) -> List[Dict[str, Any]]:
        """Process a directory of images and return pose data."""
        image_dir = self.config.input_path
        logger.info(f"Processing images in directory: {image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = sorted([
            f for f in image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process images
        all_poses = []
        
        for idx, image_path in enumerate(image_files):
            # Log progress
            if idx % 10 == 0 or idx == 0:
                progress = (idx + 1) / len(image_files) * 100
                logger.info(f"Processing image {idx+1}/{len(image_files)} ({progress:.1f}%)")
            
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
            
            # Save raw frame
            frame_filename = f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(self.config.frames_dir / frame_filename), frame)
            
            # Detect poses
            poses = self.detector.detect_poses(frame, idx)
            all_poses.extend(poses)
            
            # Draw and save overlay frame
            overlay_frame = self.detector.draw_poses(frame.copy(), poses)
            cv2.imwrite(str(self.config.overlay_dir / frame_filename), overlay_frame)
        
        logger.info(f"Image processing complete: {len(image_files)} images processed, {len(all_poses)} poses detected")
        
        return all_poses
    
    def save_results(self, poses: List[Dict[str, Any]]) -> None:
        """Save pose detection results to JSON file."""
        logger.info(f"Saving results to {self.config.output_json}")
        
        # Create output directory if it doesn't exist
        self.config.output_json.parent.mkdir(exist_ok=True, parents=True)
        
        # Save JSON
        with open(self.config.output_json, 'w') as f:
            json.dump({
                'timestamp': self.config.timestamp,
                'backend': self.backend_name,
                'input_path': str(self.config.input_path),
                'frames_dir': str(self.config.frames_dir),
                'overlay_dir': str(self.config.overlay_dir),
                'total_poses': len(poses),
                'poses': poses
            }, f, indent=2)
        
        logger.success(f"Results saved: {len(poses)} poses detected using {self.backend_name} backend")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AlphaDetect: Multi-backend pose detection tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video file')
    input_group.add_argument('--image-dir', type=str, help='Path to directory containing image files')
    
    # Output options
    parser.add_argument('--output', type=str, help='Path to output JSON file (default: outputs/pose_<timestamp>.json)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for output files')
    
    # Backend options
    parser.add_argument('--backend', type=str, default='auto', 
                        choices=['auto', 'mediapipe', 'ultralytics', 'alphapose'],
                        help='Pose detection backend to use')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='Minimum confidence threshold for detections')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    start_time = time.time()
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Check if any backends are available
        if not (MEDIAPIPE_AVAILABLE or ULTRALYTICS_AVAILABLE or ALPHAPOSE_AVAILABLE):
            logger.error("No pose detection backends available!")
            logger.error("Please install at least one of the following:")
            logger.error("  - MediaPipe: pip install mediapipe")
            logger.error("  - Ultralytics: pip install ultralytics")
            logger.error("  - AlphaPose: Follow instructions in docs/INSTALL.md")
            return 1
        
        # Initialize configuration
        config = AlphaDetectConfig(args)
        
        # Initialize pose detector
        detector = PoseDetector(config)
        
        # Process input
        if config.is_video:
            poses = detector.process_video()
        else:
            poses = detector.process_images()
        
        # Save results
        detector.save_results(poses)
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.success(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.success(f"Output saved to: {config.output_json}")
        logger.success(f"Frames saved to: {config.frames_dir}")
        logger.success(f"Overlays saved to: {config.overlay_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
