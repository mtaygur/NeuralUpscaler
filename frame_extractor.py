"""
Frame extractor for 4K HDR10 MKV files.
Extracts frames while preserving HDR metadata for post-processing.
"""

import ffmpeg
import numpy as np
from pathlib import Path
from typing import Generator, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        logger.info(f"Video info: {self.width}x{self.height} @ {self.fps:.2f}fps, {self.pix_fmt}")

    def _apply_color_processing(
        self,
        stream,
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None
    ):
        """
        Apply HDR preservation or tone mapping filters to the stream.

        Args:
            stream: ffmpeg stream object
            preserve_hdr: If True, preserve HDR metadata and color space
            tonemap_method: Tone mapping algorithm if converting to SDR

        Returns:
            Modified ffmpeg stream
        """
        if tonemap_method and not preserve_hdr:
            logger.info(f"Applying {tonemap_method} tone mapping")
            stream = stream.filter('zscale',
                                   transfer='linear',
                                   npl=100)
            stream = stream.filter('tonemap', tonemap_method)
            stream = stream.filter('zscale',
                                   transfer='bt709',
                                   matrix='bt709',
                                   primaries='bt709',
                                   range='limited')
        elif preserve_hdr:
            # Preserve HDR10 color space and transfer characteristics
            logger.info("Preserving HDR10 metadata (BT.2020, PQ transfer)")
            stream = stream.filter('zscale',
                                   matrix='bt2020nc',
                                   transfer='smpte2084',  # PQ transfer for HDR10
                                   primaries='bt2020',
                                   range='limited')
        return stream

    def extract_frames_to_files(
        self,
        output_dir: str,
        format: str = 'png',
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        pix_fmt: str = 'rgb48be',  # 16-bit RGB for HDR preservation
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None  # 'hable', 'reinhard', 'mobius', etc.
    ) -> None:
        """
        Extract frames and save to individual files.

        Args:
            output_dir: Directory to save extracted frames
            format: Output format ('png', 'tiff', 'exr')
            start_frame: Starting frame number (None for beginning)
            end_frame: Ending frame number (None for end of video)
            pix_fmt: Pixel format for output (rgb48be for 16-bit HDR)
            preserve_hdr: If True, preserve HDR metadata and color space (no tone mapping)
            tonemap_method: Tone mapping algorithm if converting to SDR ('hable', 'reinhard', 'mobius')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        stream = ffmpeg.input(str(self.input_file))

        # Add frame range if specified
        if start_frame is not None:
            stream = stream.filter('select', f'gte(n,{start_frame})')
        if end_frame is not None:
            stream = stream.filter('select', f'lte(n,{end_frame})')

        # Apply color processing (HDR preservation or tone mapping)
        stream = self._apply_color_processing(stream, preserve_hdr, tonemap_method)

        # Output with HDR-compatible pixel format
        output_pattern = str(output_path / f'frame_%06d.{format}')
        stream = ffmpeg.output(
            stream,
            output_pattern,
            pix_fmt=pix_fmt,
            **{'qscale:v': 1}  # High quality for PNG
        )

        logger.info(f"Extracting frames to {output_dir}")
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        logger.info("Frame extraction complete")

    def extract_frames_generator(
        self,
        pix_fmt: str = 'rgb48le',  # 16-bit RGB for HDR
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Extract frames as numpy arrays using a generator (memory efficient).

        Args:
            pix_fmt: Pixel format ('rgb48le' for 16-bit RGB, 'rgb24' for 8-bit)
            start_time: Start time in seconds
            end_time: End time in seconds
            preserve_hdr: If True, preserve HDR metadata and color space (no tone mapping)
            tonemap_method: Tone mapping algorithm if converting to SDR ('hable', 'reinhard', 'mobius')

        Yields:
            numpy.ndarray: Frame data as numpy array
        """
        # Determine bytes per pixel based on format
        if 'rgb48' in pix_fmt or 'rgb16' in pix_fmt:
            bytes_per_pixel = 6  # 16-bit per channel, 3 channels
            dtype = np.uint16
        else:
            bytes_per_pixel = 3  # 8-bit per channel, 3 channels
            dtype = np.uint8

        # Build ffmpeg command
        stream = ffmpeg.input(str(self.input_file))

        # Add time range if specified
        if start_time is not None:
            stream = stream.filter('trim', start=start_time)
        if end_time is not None:
            stream = stream.filter('trim', end=end_time)

        # Apply color processing (HDR preservation or tone mapping)
        stream = self._apply_color_processing(stream, preserve_hdr, tonemap_method)

        # Output to pipe
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt=pix_fmt)

        process = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=True)

        frame_size = self.width * self.height * bytes_per_pixel
        frame_count = 0

        try:
            while True:
                # Read frame data
                in_bytes = process.stdout.read(frame_size)

                if not in_bytes:
                    break

                if len(in_bytes) != frame_size:
                    logger.warning(f"Incomplete frame {frame_count}, skipping")
                    break

                # Convert to numpy array
                frame = np.frombuffer(in_bytes, dtype=dtype).reshape([self.height, self.width, 3])

                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

                yield frame

        finally:
            process.stdout.close()
            process.wait()
            logger.info(f"Total frames processed: {frame_count}")



def example_usage():
    """Example usage of the HDRFrameExtractor."""

    # Initialize extractor
    extractor = HDRFrameExtractor('input_video.mkv')

    # Method 1: Extract frames to files with HDR preservation (default)
    # Preserves BT.2020 color space and PQ (SMPTE ST 2084) transfer function
    # extractor.extract_frames_to_files(
    #     output_dir='frames_hdr',
    #     format='png',
    #     pix_fmt='rgb48be',  # 16-bit RGB for HDR
    #     preserve_hdr=True   # Keeps HDR10 metadata
    # )

    # Method 2: Extract with tone mapping to SDR
    # extractor.extract_frames_to_files(
    #     output_dir='frames_sdr',
    #     format='png',
    #     pix_fmt='rgb24',  # 8-bit RGB for SDR
    #     preserve_hdr=False,
    #     tonemap_method='hable'  # Options: 'hable', 'reinhard', 'mobius'
    # )

    # Method 3: Stream frames one at a time (MEMORY EFFICIENT)
    # Only one frame is in memory at a time (~50MB for 4K HDR)
    # Perfect for processing large 50GB+ video files
    for i, frame in enumerate(extractor.extract_frames_generator(
        pix_fmt='rgb48le',
        preserve_hdr=True  # Preserve HDR10 color space (BT.2020, PQ)
    )):
        # Process frame here - frame contains LINEAR HDR data in BT.2020 color space
        print(f"Processing frame {i}: shape={frame.shape}, dtype={frame.dtype}")
        # Values range from 0-65535 for 16-bit HDR (maps to 0.0-1.0+ in linear space)

        # Example: Apply your post-processing to this single frame
        # processed_frame = your_postprocessing_function(frame)
        # save_processed_frame(processed_frame, f'output_{i:06d}.png')

        # Break after a few frames for demonstration
        if i >= 5:
            break
