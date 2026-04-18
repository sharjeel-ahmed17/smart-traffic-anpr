import cv2
import os
from typing import Dict, Any, Optional


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video metadata including fps, resolution, duration, and total frames.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration_seconds = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_seconds": round(duration_seconds, 2)
    }


def extract_frames(
    video_path: str,
    output_dir: str = "data/images",
    frame_interval: int = 30,
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> list:
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30)
        start_frame: Starting frame index (default: 0)
        end_frame: Ending frame index (default: None = end of video)

    Returns:
        List of paths to saved frame images
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_frames = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if end_frame is not None and frame_count >= end_frame:
            break

        if frame_count >= start_frame and (frame_count - start_frame) % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Extracted {saved_count} frames from {video_path}")
    print(f"Saved to: {output_dir}")

    return saved_frames


def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Load a video file and return VideoCapture object.

    Args:
        video_path: Path to the video file

    Returns:
        cv2.VideoCapture object

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    return cap


def list_videos(directory: str = "data/videos") -> list:
    """
    List all video files in a directory.

    Args:
        directory: Directory to search for video files

    Returns:
        List of video file paths
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

    if not os.path.exists(directory):
        return []

    videos = []
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1].lower() in video_extensions:
            videos.append(os.path.join(directory, filename))

    return sorted(videos)


if __name__ == "__main__":
    videos = list_videos()

    if not videos:
        print("No videos found in data/videos/")
    else:
        print(f"Found {len(videos)} video(s):")
        for video in videos:
            print(f"\n  - {video}")
            info = get_video_info(video)
            print(f"    Resolution: {info['width']}x{info['height']}")
            print(f"    FPS: {info['fps']}")
            print(f"    Duration: {info['duration_seconds']}s")
            print(f"    Total frames: {info['total_frames']}")
