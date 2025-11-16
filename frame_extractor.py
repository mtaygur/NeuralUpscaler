"""
Frame extractor for 4K HDR10 MKV files.
Extracts frames while preserving HDR metadata for post-processing.
"""

import logging
from pathlib import Path
from typing import Optional

import ffmpeg

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
