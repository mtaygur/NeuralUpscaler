"""
NeuralUpscaler - Main Entry Point

Example usage of frame extraction with quality filtering for HDR video upscaling.
"""

import sys
from pathlib import Path
from frame_extractor import HDRFrameExtractor
from frame_quality_filter import (
    FrameQualityFilter,
    create_default_filter,
    create_strict_filter,
    create_lenient_filter,
    QualityThresholds
)


def filter_and_save_frames(
    input_video: str,
    output_dir: str,
    quality_preset: str = "default",
    verbose: bool = True,
    format: str = "png",
    start_time: float = None,
    end_time: float = None
):
    """
    Extract frames from video, filter by quality, and save high-quality frames.

    Args:
        input_video: Path to input MKV file
        output_dir: Directory to save filtered frames
        quality_preset: Quality filter preset ("default", "strict", "lenient", or "custom")
        verbose: Print quality metrics for each frame
        format: Output format (png, tiff, exr)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize frame extractor
    print(f"Initializing HDR frame extractor for: {input_video}")
    extractor = HDRFrameExtractor(input_video)

    # Print video info
    print(f"\nVideo Information:")
    print(f"  Resolution: {extractor.width}x{extractor.height}")
    print(f"  FPS: {extractor.fps}")
    print(f"  Pixel Format: {extractor.pix_fmt}")

    # Create quality filter based on preset
    print(f"\nCreating quality filter (preset: {quality_preset})")
    if quality_preset == "strict":
        quality_filter = create_strict_filter()
    elif quality_preset == "lenient":
        quality_filter = create_lenient_filter()
    elif quality_preset == "custom":
        # Example of custom thresholds
        custom_thresholds = QualityThresholds(
            min_blur_score=150.0,
            max_brisque_score=40.0,
            min_brightness_16bit=6000.0,
            max_brightness_16bit=58000.0,
            min_contrast_16bit=4000.0,
        )
        quality_filter = FrameQualityFilter(custom_thresholds)
    else:  # default
        quality_filter = create_default_filter()

    # Extract frames using generator
    print(f"\nExtracting and filtering frames...")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {format}")

    frames_generator = extractor.extract_frames_generator(
        start_time=start_time,
        end_time=end_time
    )

    # Filter frames and save
    frame_count = 0
    saved_count = 0

    for frame in quality_filter.filter_frames(frames_generator, verbose=verbose):
        # Save the frame
        frame_filename = output_path / f"frame_{saved_count:06d}.{format}"

        # Convert frame to appropriate format for saving
        if format == "png":
            import cv2
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_filename), bgr_frame)
        elif format in ["tiff", "tif"]:
            import cv2
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_filename), bgr_frame)
        else:
            # For other formats, you might need additional libraries
            print(f"Warning: Format {format} might require additional handling")

        saved_count += 1

        if verbose:
            print(f"  Saved: {frame_filename}")

    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"  High-quality frames saved: {saved_count}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def analyze_frame_quality(input_video: str, sample_rate: int = 30):
    """
    Analyze frame quality without saving frames.

    Useful for determining appropriate quality thresholds.

    Args:
        input_video: Path to input MKV file
        sample_rate: Analyze every Nth frame (default: 30, ~1 per second for 30fps video)
    """
    print(f"Analyzing frame quality for: {input_video}")

    # Initialize frame extractor and quality filter
    extractor = HDRFrameExtractor(input_video)
    quality_filter = create_default_filter()

    print(f"\nVideo Information:")
    print(f"  Resolution: {extractor.width}x{extractor.height}")
    print(f"  FPS: {extractor.fps}")

    # Extract and analyze frames
    frames_generator = extractor.extract_frames_generator()

    frame_count = 0
    analyzed_count = 0
    blur_scores = []
    brisque_scores = []
    brightness_values = []
    contrast_values = []

    for frame in frames_generator:
        frame_count += 1

        # Only analyze every Nth frame
        if frame_count % sample_rate != 0:
            continue

        analyzed_count += 1
        metrics = quality_filter.assess_quality(frame)

        blur_scores.append(metrics.blur_score)
        brisque_scores.append(metrics.brisque_score)
        brightness_values.append(metrics.mean_brightness)
        contrast_values.append(metrics.contrast)

        print(f"\nFrame {frame_count}:")
        print(f"  Blur: {metrics.blur_score:.2f}")
        print(f"  BRISQUE: {metrics.brisque_score:.2f}")
        print(f"  Brightness: {metrics.mean_brightness:.2f}")
        print(f"  Contrast: {metrics.contrast:.2f}")
        print(f"  Quality: {'PASS ✓' if metrics.passes_all else 'FAIL ❌'}")

    # Print summary statistics
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Quality Analysis Summary ({analyzed_count} frames analyzed):")
    print(f"  Blur Score - Min: {min(blur_scores):.2f}, Max: {max(blur_scores):.2f}, Mean: {np.mean(blur_scores):.2f}")
    print(f"  BRISQUE - Min: {min(brisque_scores):.2f}, Max: {max(brisque_scores):.2f}, Mean: {np.mean(brisque_scores):.2f}")
    print(f"  Brightness - Min: {min(brightness_values):.2f}, Max: {max(brightness_values):.2f}, Mean: {np.mean(brightness_values):.2f}")
    print(f"  Contrast - Min: {min(contrast_values):.2f}, Max: {max(contrast_values):.2f}, Mean: {np.mean(contrast_values):.2f}")
    print(f"{'='*60}")


def main():
    """Main entry point with example usage"""

    # Example 1: Filter and save high-quality frames with default settings
    # Uncomment and modify the path to your video file
    """
    filter_and_save_frames(
        input_video="path/to/your/video.mkv",
        output_dir="output/high_quality_frames",
        quality_preset="default",
        verbose=True
    )
    """

    # Example 2: Use strict quality filtering
    """
    filter_and_save_frames(
        input_video="path/to/your/video.mkv",
        output_dir="output/strict_quality_frames",
        quality_preset="strict",
        verbose=True
    )
    """

    # Example 3: Analyze frame quality to determine thresholds
    """
    analyze_frame_quality(
        input_video="path/to/your/video.mkv",
        sample_rate=30  # Analyze every 30th frame
    )
    """

    # Example 4: Filter specific time range
    """
    filter_and_save_frames(
        input_video="path/to/your/video.mkv",
        output_dir="output/scene_frames",
        quality_preset="default",
        start_time=60.0,  # Start at 1 minute
        end_time=120.0,   # End at 2 minutes
        verbose=True
    )
    """

    print("NeuralUpscaler - Frame Quality Filtering")
    print("=" * 60)
    print("\nTo use this tool, uncomment one of the examples in main() function")
    print("or create your own processing pipeline.")
    print("\nAvailable quality presets:")
    print("  - 'default': Balanced quality filtering")
    print("  - 'strict': Aggressive filtering for highest quality")
    print("  - 'lenient': Permissive filtering to keep more frames")
    print("  - 'custom': Define your own QualityThresholds")
    print("\nQuality Checks:")
    print("  1. Blur detection (Laplacian variance)")
    print("  2. Compression artifacts (BRISQUE score)")
    print("  3. Brightness issues (too dark/too bright)")
    print("  4. Low contrast (single-tone images)")
    print("=" * 60)


if __name__ == '__main__':
    main()
