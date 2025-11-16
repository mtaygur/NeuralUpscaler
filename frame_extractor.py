"""
Frame extractor for 4K HDR10 MKV files.
Extracts frames while preserving HDR metadata for post-processing.
"""

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import ffmpeg
import numpy as np

from frame_quality_filter import FrameQualityFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed values for subprocess command parameters (prevents injection)
ALLOWED_TONEMAP_METHODS = {'hable', 'reinhard', 'mobius'}
ALLOWED_PIXEL_FORMATS = {'rgb48be', 'rgb48le', 'rgb24'}  # RGB formats only, 48-bit uses 2 bytes/channel
ALLOWED_OUTPUT_FORMATS = {'png', 'tiff', 'exr'}


class HDRFrameExtractor:
    """Extract frames from 4K HDR10 MKV files."""

    def __init__(self, input_file: str):
        """
        Initialize the frame extractor.

        Args:
            input_file: Path to the input MKV file
        """
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Get video information
        self.probe = ffmpeg.probe(str(self.input_file))
        self.video_stream = next(
            (s for s in self.probe['streams'] if s['codec_type'] == 'video'),
            None
        )

        if not self.video_stream:
            raise ValueError("No video stream found in file")

        self.width = int(self.video_stream['width'])
        self.height = int(self.video_stream['height'])
        self.fps = eval(self.video_stream['r_frame_rate'])  # e.g., "24000/1001"
        self.pix_fmt = self.video_stream.get('pix_fmt', 'yuv420p10le')

        # Cache file size and duration
        self._file_size_bytes = self.input_file.stat().st_size
        self._duration = self._get_duration_from_probe()

        logger.info(f"Video info: {self.width}x{self.height} @ {self.fps:.2f}fps, {self.pix_fmt}")
        logger.info(f"File size: {self._file_size_bytes / (1024**3):.2f} GB, Duration: {self._duration:.2f}s")

    def _get_duration_from_probe(self) -> float:
        """Extract video duration from probe data."""
        if 'duration' in self.video_stream:
            return float(self.video_stream['duration'])
        if 'format' in self.probe and 'duration' in self.probe['format']:
            return float(self.probe['format']['duration'])
        if 'nb_frames' in self.video_stream:
            return int(self.video_stream['nb_frames']) / self.fps
        raise ValueError("Cannot determine video duration from metadata")

    def _apply_color_processing(
        self,
        stream,
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None
    ):
        """Apply HDR preservation or tone mapping filters to the stream."""
        if tonemap_method and not preserve_hdr:
            logger.info(f"Applying {tonemap_method} tone mapping")
            stream = stream.filter('zscale', transfer='linear', npl=100)
            stream = stream.filter('tonemap', tonemap_method)
            stream = stream.filter('zscale',
                                   transfer='bt709',
                                   matrix='bt709',
                                   primaries='bt709',
                                   range='limited')
        elif preserve_hdr:
            logger.info("Preserving HDR10 metadata (BT.2020, PQ transfer)")
            stream = stream.filter('zscale',
                                   matrix='bt2020nc',
                                   transfer='smpte2084',
                                   primaries='bt2020',
                                   range='limited')
        return stream

    def extract_frames_to_files(
        self,
        output_dir: str,
        output_format: str = 'png',
        pix_fmt: str = 'rgb48be',
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None
    ) -> None:
        """
        Extract all frames from video and save to individual files.

        Args:
            output_dir: Directory to save extracted frames
            output_format: Output format ('png', 'tiff', 'exr')
            pix_fmt: Pixel format for output (rgb48be for 16-bit HDR)
            preserve_hdr: If True, preserve HDR metadata and color space
            tonemap_method: Tone mapping algorithm if converting to SDR
                           ('hable', 'reinhard', 'mobius')
        """
        # Validate parameters against allowed values
        if output_format not in ALLOWED_OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {ALLOWED_OUTPUT_FORMATS}")
        if pix_fmt not in ALLOWED_PIXEL_FORMATS:
            raise ValueError(f"pix_fmt must be one of {ALLOWED_PIXEL_FORMATS}")
        if tonemap_method is not None and tonemap_method not in ALLOWED_TONEMAP_METHODS:
            raise ValueError(f"tonemap_method must be one of {ALLOWED_TONEMAP_METHODS}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting frames to {output_dir}")
        stream = ffmpeg.input(str(self.input_file))
        stream = self._apply_color_processing(stream, preserve_hdr, tonemap_method)

        output_pattern = str(output_path / f'frame_%06d.{output_format}')
        stream = ffmpeg.output(
            stream,
            output_pattern,
            pix_fmt=pix_fmt,
            **{'qscale:v': 1}
        )

        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        logger.info("Frame extraction complete")

    def extract_frames_with_quality_filter(
        self,
        output_dir: str,
        quality_filter: FrameQualityFilter,
        output_format: str = 'png',
        pix_fmt: str = 'rgb48be',
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None,
        batch_size: int = 20,
        num_workers: Optional[int] = None
    ) -> int:
        """
        Extract frames with quality filtering using streaming and parallel processing.

        Frames are streamed through memory, assessed for quality in parallel,
        and only frames passing quality gates are written to disk.

        Args:
            output_dir: Directory to save extracted frames
            quality_filter: FrameQualityFilter instance for quality assessment
            output_format: Output format ('png', 'tiff', 'exr')
            pix_fmt: Pixel format for output (rgb48be for 16-bit HDR)
            preserve_hdr: If True, preserve HDR metadata and color space
            tonemap_method: Tone mapping algorithm if converting to SDR
            batch_size: Number of frames to process in each batch
            num_workers: Number of parallel workers (default: CPU count - 1)

        Returns:
            Number of frames that passed quality filtering
        """
        import os

        # Validate parameters against allowed values
        if output_format not in ALLOWED_OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {ALLOWED_OUTPUT_FORMATS}")
        if pix_fmt not in ALLOWED_PIXEL_FORMATS:
            raise ValueError(f"pix_fmt must be one of {ALLOWED_PIXEL_FORMATS}")
        if tonemap_method is not None and tonemap_method not in ALLOWED_TONEMAP_METHODS:
            raise ValueError(f"tonemap_method must be one of {ALLOWED_TONEMAP_METHODS}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)

        # Build FFmpeg command for raw video output
        cmd = self._build_ffmpeg_pipe_command(preserve_hdr, tonemap_method, pix_fmt)

        logger.info("Starting streaming extraction with quality filtering")
        logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")

        # Determine frame size based on pixel format
        is_16bit = pix_fmt in ('rgb48be', 'rgb48le')
        bytes_per_channel = 2 if is_16bit else 1
        frame_size = self.width * self.height * 3 * bytes_per_channel
        dtype = np.uint16 if is_16bit else np.uint8

        # Start FFmpeg process
        # shell=False ensures arguments are passed directly to ffmpeg without shell interpretation,
        # preventing command injection attacks
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_size * batch_size,
            shell=False
        )

        frame_num = 0
        kept_count = 0
        total_processed = 0

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                while True:
                    # Read batch of frames from pipe
                    batch_bytes = process.stdout.read(frame_size * batch_size)

                    if len(batch_bytes) == 0:
                        break

                    # Handle partial batch at end of video
                    actual_frames = len(batch_bytes) // frame_size
                    if actual_frames == 0:
                        break

                    # Truncate to complete frames only
                    batch_bytes = batch_bytes[:actual_frames * frame_size]

                    # Convert to numpy array
                    batch_array = np.frombuffer(batch_bytes, dtype=dtype)
                    batch_array = batch_array.reshape((actual_frames, self.height, self.width, 3))

                    # Submit all frames in batch to worker pool
                    futures = []
                    for i in range(actual_frames):
                        frame = batch_array[i].copy()  # Copy for worker
                        future = executor.submit(quality_filter.should_keep_frame, frame)
                        futures.append((frame_num + i, frame, future))

                    # Collect results and write passing frames
                    for frame, future in futures:
                        passes_quality = future.result()
                        if passes_quality:
                            output_file = output_path / f'frame_{kept_count + 1:06d}.{output_format}'
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(output_file), frame_bgr)
                            kept_count += 1

                    frame_num += actual_frames
                    total_processed += actual_frames

                    # Progress logging every 100 frames
                    if total_processed % 100 < batch_size:
                        logger.info(f"Processed {total_processed} frames, kept {kept_count} "
                                   f"({kept_count/total_processed*100:.1f}%)")

        finally:
            process.stdout.close()
            process.stderr.close()
            process.wait()

        logger.info(f"Extraction complete. Processed {total_processed} frames, "
                   f"kept {kept_count} ({kept_count/total_processed*100:.1f}% pass rate)")

        return kept_count

    def _build_ffmpeg_pipe_command(
        self,
        preserve_hdr: bool,
        tonemap_method: Optional[str],
        pix_fmt: str
    ) -> list:
        """
        Build FFmpeg command for piping raw video output.

        Args:
            preserve_hdr: If True, preserve HDR metadata
            tonemap_method: Tone mapping algorithm if converting to SDR
            pix_fmt: Pixel format for output

        Returns:
            List of command arguments for subprocess
        """
        cmd = ['ffmpeg', '-i', str(self.input_file)]

        # Build filter chain
        filters = []

        if tonemap_method and not preserve_hdr:
            logger.info(f"Applying {tonemap_method} tone mapping")
            filters.extend([
                'zscale=transfer=linear:npl=100',
                f'tonemap={tonemap_method}',
                'zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=limited'
            ])
        elif preserve_hdr:
            logger.info("Preserving HDR10 metadata (BT.2020, PQ transfer)")
            filters.append(
                'zscale=matrix=bt2020nc:transfer=smpte2084:primaries=bt2020:range=limited'
            )

        if filters:
            cmd.extend(['-vf', ','.join(filters)])

        # Output to pipe as raw video
        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', pix_fmt,
            'pipe:1'
        ])

        return cmd
