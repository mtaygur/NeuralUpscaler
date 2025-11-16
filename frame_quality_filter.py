"""
Frame Quality Filter Module

This module provides quality assessment and filtering for video frames.
It filters out low-quality frames based on multiple criteria:
1. Blur detection (Laplacian variance)
2. Compression artifacts (BRISQUE score)
3. Brightness issues (overexposed/underexposed)
4. Low contrast (single-tone images)
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QualityThresholds:
    """Thresholds for frame quality assessment"""
    # Blur detection - lower values indicate more blur
    min_blur_score: float = 100.0  # Laplacian variance threshold

    # BRISQUE score - higher values indicate worse quality (0-100 scale)
    max_brisque_score: float = 50.0  # Maximum acceptable BRISQUE score

    # Brightness thresholds (0-255 for 8-bit, 0-65535 for 16-bit)
    min_brightness_8bit: float = 20.0  # Too dark threshold
    max_brightness_8bit: float = 235.0  # Too bright threshold
    min_brightness_16bit: float = 5120.0  # Too dark (20/255 * 65535)
    max_brightness_16bit: float = 60395.0  # Too bright (235/255 * 65535)

    # Contrast threshold - standard deviation of pixel values
    min_contrast_8bit: float = 15.0  # Minimum std dev for 8-bit
    min_contrast_16bit: float = 3855.0  # Minimum std dev for 16-bit (15/255 * 65535)


@dataclass
class QualityMetrics:
    """Container for frame quality metrics"""
    blur_score: float  # Laplacian variance
    brisque_score: float  # BRISQUE quality score
    mean_brightness: float  # Mean pixel value
    contrast: float  # Standard deviation of pixels
    is_blurry: bool
    has_artifacts: bool
    is_too_bright: bool
    is_too_dark: bool
    is_low_contrast: bool
    passes_all: bool  # True if frame passes all quality checks


class FrameQualityFilter:
    """
    Frame quality assessment and filtering.

    Supports both 8-bit and 16-bit images (HDR).
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize the quality filter.

        Args:
            thresholds: Custom quality thresholds. If None, uses defaults.
        """
        self.thresholds = thresholds or QualityThresholds()

        # Initialize BRISQUE quality scorer
        # Note: OpenCV's BRISQUE implementation requires trained model
        try:
            self.brisque = cv2.quality.QualityBRISQUE_create()
        except AttributeError:
            # Fallback if opencv-contrib-python is not installed
            # We'll use a simplified quality metric instead
            self.brisque = None
            print("Warning: BRISQUE not available. Install opencv-contrib-python for full quality assessment.")

    def _normalize_to_8bit(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to 8-bit for processing.

        Args:
            frame: Input frame (8-bit or 16-bit)

        Returns:
            8-bit normalized frame
        """
        if frame.dtype == np.uint16:
            # Convert 16-bit to 8-bit by scaling
            return (frame / 256).astype(np.uint8)
        elif frame.dtype == np.uint8:
            return frame
        else:
            # Handle float types
            if frame.max() <= 1.0:
                return (frame * 255).astype(np.uint8)
            else:
                return frame.astype(np.uint8)

    def detect_blur(self, frame: np.ndarray) -> float:
        """
        Detect blur using Laplacian variance.

        Higher values indicate sharper images.
        Typical threshold: < 100 is blurry, > 100 is sharp.

        Args:
            frame: Input frame (RGB)

        Returns:
            Blur score (Laplacian variance)
        """
        # Convert to 8-bit grayscale for blur detection
        frame_8bit = self._normalize_to_8bit(frame)
        gray = cv2.cvtColor(frame_8bit, cv2.COLOR_RGB2GRAY)

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance

    def detect_artifacts_brisque(self, frame: np.ndarray) -> float:
        """
        Detect compression artifacts using BRISQUE.

        BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
        scores range from 0-100, where lower is better quality.
        Typical threshold: > 50 indicates poor quality.

        Args:
            frame: Input frame (RGB)

        Returns:
            BRISQUE score (0-100, lower is better)
        """
        if self.brisque is None:
            # Fallback: use a simplified artifact detection based on high-frequency noise
            return self._detect_artifacts_fallback(frame)

        # Convert to 8-bit BGR for BRISQUE (OpenCV format)
        frame_8bit = self._normalize_to_8bit(frame)
        bgr = cv2.cvtColor(frame_8bit, cv2.COLOR_RGB2BGR)

        # Compute BRISQUE score
        score = self.brisque.compute(bgr)[0]
        return score

    def _detect_artifacts_fallback(self, frame: np.ndarray) -> float:
        """
        Fallback artifact detection using high-frequency noise analysis.

        Args:
            frame: Input frame (RGB)

        Returns:
            Estimated quality score (0-100, lower is better)
        """
        # Convert to 8-bit grayscale
        frame_8bit = self._normalize_to_8bit(frame)
        gray = cv2.cvtColor(frame_8bit, cv2.COLOR_RGB2GRAY)

        # Calculate high-frequency noise using gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # High variance in gradients can indicate noise/artifacts
        # Normalize to 0-100 scale (empirically derived)
        noise_score = min(100, gradient_magnitude.std() * 2)

        return noise_score

    def check_brightness(self, frame: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Check if frame is too bright or too dark.

        Args:
            frame: Input frame (RGB)

        Returns:
            Tuple of (mean_brightness, is_too_dark, is_too_bright)
        """
        # Calculate mean brightness across all channels
        mean_brightness = frame.mean()

        # Determine thresholds based on bit depth
        if frame.dtype == np.uint16:
            is_too_dark = mean_brightness < self.thresholds.min_brightness_16bit
            is_too_bright = mean_brightness > self.thresholds.max_brightness_16bit
        else:
            # Treat as 8-bit
            if frame.max() <= 255:
                is_too_dark = mean_brightness < self.thresholds.min_brightness_8bit
                is_too_bright = mean_brightness > self.thresholds.max_brightness_8bit
            else:
                # Scale thresholds for 16-bit
                is_too_dark = mean_brightness < self.thresholds.min_brightness_16bit
                is_too_bright = mean_brightness > self.thresholds.max_brightness_16bit

        return mean_brightness, is_too_dark, is_too_bright

    def check_contrast(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Check if frame has low contrast (single-tone).

        Uses standard deviation as a measure of contrast.

        Args:
            frame: Input frame (RGB)

        Returns:
            Tuple of (contrast_score, is_low_contrast)
        """
        # Calculate standard deviation across all channels
        contrast = frame.std()

        # Determine threshold based on bit depth
        if frame.dtype == np.uint16:
            is_low_contrast = contrast < self.thresholds.min_contrast_16bit
        else:
            if frame.max() <= 255:
                is_low_contrast = contrast < self.thresholds.min_contrast_8bit
            else:
                is_low_contrast = contrast < self.thresholds.min_contrast_16bit

        return contrast, is_low_contrast

    def assess_quality(self, frame: np.ndarray) -> QualityMetrics:
        """
        Perform comprehensive quality assessment on a frame.

        Args:
            frame: Input frame (RGB, 8-bit or 16-bit)

        Returns:
            QualityMetrics object with all quality measurements
        """
        # Detect blur
        blur_score = self.detect_blur(frame)
        is_blurry = blur_score < self.thresholds.min_blur_score

        # Detect artifacts
        brisque_score = self.detect_artifacts_brisque(frame)
        has_artifacts = brisque_score > self.thresholds.max_brisque_score

        # Check brightness
        mean_brightness, is_too_dark, is_too_bright = self.check_brightness(frame)

        # Check contrast
        contrast, is_low_contrast = self.check_contrast(frame)

        # Determine if frame passes all checks
        passes_all = not (is_blurry or has_artifacts or is_too_bright or
                         is_too_dark or is_low_contrast)

        return QualityMetrics(
            blur_score=blur_score,
            brisque_score=brisque_score,
            mean_brightness=mean_brightness,
            contrast=contrast,
            is_blurry=is_blurry,
            has_artifacts=has_artifacts,
            is_too_bright=is_too_bright,
            is_too_dark=is_too_dark,
            is_low_contrast=is_low_contrast,
            passes_all=passes_all
        )

    def should_keep_frame(self, frame: np.ndarray) -> bool:
        """
        Quick check if frame should be kept.

        Args:
            frame: Input frame (RGB)

        Returns:
            True if frame passes quality checks, False otherwise
        """
        metrics = self.assess_quality(frame)
        return metrics.passes_all

    def filter_frames(self, frames_generator, verbose: bool = False):
        """
        Filter frames from a generator, yielding only high-quality frames.

        Args:
            frames_generator: Generator yielding frames
            verbose: If True, print quality metrics for each frame

        Yields:
            High-quality frames that pass all quality checks
        """
        total_frames = 0
        kept_frames = 0

        for frame in frames_generator:
            total_frames += 1
            metrics = self.assess_quality(frame)

            if verbose:
                print(f"\nFrame {total_frames}:")
                print(f"  Blur score: {metrics.blur_score:.2f} {'❌ BLURRY' if metrics.is_blurry else '✓'}")
                print(f"  BRISQUE score: {metrics.brisque_score:.2f} {'❌ ARTIFACTS' if metrics.has_artifacts else '✓'}")
                print(f"  Brightness: {metrics.mean_brightness:.2f} {'❌ TOO DARK' if metrics.is_too_dark else '❌ TOO BRIGHT' if metrics.is_too_bright else '✓'}")
                print(f"  Contrast: {metrics.contrast:.2f} {'❌ LOW CONTRAST' if metrics.is_low_contrast else '✓'}")
                print(f"  Result: {'✓ KEEP' if metrics.passes_all else '❌ DISCARD'}")

            if metrics.passes_all:
                kept_frames += 1
                yield frame

        if verbose or total_frames > 0:
            print(f"\n{'='*60}")
            print(f"Quality Filtering Summary:")
            print(f"  Total frames: {total_frames}")
            print(f"  Kept frames: {kept_frames}")
            print(f"  Discarded frames: {total_frames - kept_frames}")
            print(f"  Keep rate: {kept_frames/total_frames*100:.1f}%")
            print(f"{'='*60}")


def create_default_filter() -> FrameQualityFilter:
    """
    Create a frame quality filter with default thresholds.

    Returns:
        FrameQualityFilter with default settings
    """
    return FrameQualityFilter()


def create_strict_filter() -> FrameQualityFilter:
    """
    Create a frame quality filter with strict thresholds.

    Returns:
        FrameQualityFilter with stricter quality requirements
    """
    strict_thresholds = QualityThresholds(
        min_blur_score=200.0,  # More strict blur detection
        max_brisque_score=30.0,  # Lower tolerance for artifacts
        min_brightness_8bit=30.0,
        max_brightness_8bit=225.0,
        min_brightness_16bit=7710.0,
        max_brightness_16bit=57855.0,
        min_contrast_8bit=25.0,
        min_contrast_16bit=6425.0,
    )
    return FrameQualityFilter(strict_thresholds)


def create_lenient_filter() -> FrameQualityFilter:
    """
    Create a frame quality filter with lenient thresholds.

    Returns:
        FrameQualityFilter with more permissive quality requirements
    """
    lenient_thresholds = QualityThresholds(
        min_blur_score=50.0,  # More lenient blur detection
        max_brisque_score=70.0,  # Higher tolerance for artifacts
        min_brightness_8bit=10.0,
        max_brightness_8bit=245.0,
        min_brightness_16bit=2570.0,
        max_brightness_16bit=62965.0,
        min_contrast_8bit=10.0,
        min_contrast_16bit=2570.0,
    )
    return FrameQualityFilter(lenient_thresholds)
