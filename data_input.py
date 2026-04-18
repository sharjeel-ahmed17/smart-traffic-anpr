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


def extract_frames_batch(
    video_dir: str = "data/videos",
    output_base_dir: str = "data/images",
    frame_interval: int = 30
) -> Dict[str, list]:
    """
    Extract frames from all videos in a directory.

    Args:
        video_dir: Directory containing video files
        output_base_dir: Base directory for output frames
        frame_interval: Extract every Nth frame

    Returns:
        Dictionary mapping video names to lists of saved frame paths
    """
    videos = list_videos(video_dir)
    results = {}

    for video_path in videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, video_name)
        saved_frames = extract_frames(video_path, output_dir, frame_interval)
        results[video_name] = saved_frames

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ANPR Data Input - Load CCTV footage and extract frames"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available videos")
    list_parser.add_argument(
        "--dir", default="data/videos", help="Directory to search for videos"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Get video metadata")
    info_parser.add_argument("video", help="Path to video file")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract frames from video")
    extract_parser.add_argument("video", nargs="?", help="Path to video file (optional)")
    extract_parser.add_argument(
        "--all", action="store_true", help="Extract frames from all videos in data/videos/"
    )
    extract_parser.add_argument(
        "--output", "-o", default="data/images", help="Output directory for frames"
    )
    extract_parser.add_argument(
        "--interval", "-i", type=int, default=30, help="Extract every Nth frame (default: 30)"
    )
    extract_parser.add_argument(
        "--start", type=int, default=0, help="Starting frame index"
    )
    extract_parser.add_argument(
        "--end", type=int, help="Ending frame index"
    )

    args = parser.parse_args()

    if args.command == "list":
        videos = list_videos(args.dir)
        if not videos:
            print(f"No videos found in {args.dir}/")
        else:
            print(f"Found {len(videos)} video(s):")
            for video in videos:
                print(f"  - {video}")

    elif args.command == "info":
        info = get_video_info(args.video)
        print(f"Video: {args.video}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}")
        print(f"  Duration: {info['duration_seconds']}s ({info['duration_seconds']/60:.2f} min)")
        print(f"  Total frames: {info['total_frames']}")

    elif args.command == "extract":
        if args.all:
            print("Extracting frames from all videos...")
            results = extract_frames_batch("data/videos", args.output, args.interval)
            for video_name, frames in results.items():
                print(f"  {video_name}: {len(frames)} frames extracted")
        elif args.video:
            extract_frames(
                args.video,
                args.output,
                args.interval,
                args.start,
                args.end
            )
        else:
            print("Error: Specify a video file or use --all flag")
            parser.print_help()

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python data_input.py list")
        print("  python data_input.py info data/videos/traffic1.mp4")
        print("  python data_input.py extract data/videos/traffic1.mp4")
        print("  python data_input.py extract --all --interval 60")
