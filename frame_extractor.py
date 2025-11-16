"""
Frame extractor for 4K HDR10 MKV files.
Extracts frames while preserving HDR metadata for post-processing.
"""

import logging
from pathlib import Path
from typing import Optional

import ffmpeg
import numpy as np

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

    def _get_chunk_boundaries(self, chunk_size_gb: float) -> list[dict]:
        """Calculate chunk boundaries based on file size."""
        chunk_size_bytes = chunk_size_gb * (1024 ** 3)
        num_chunks = int(np.ceil(self._file_size_bytes / chunk_size_bytes))

        chunks = []
        for i in range(num_chunks):
            start_ratio = (i * chunk_size_bytes) / self._file_size_bytes
            end_ratio = min(((i + 1) * chunk_size_bytes) / self._file_size_bytes, 1.0)

            chunks.append({
                'chunk_index': i,
                'start_time': start_ratio * self._duration,
                'end_time': end_ratio * self._duration,
            })

        return chunks

    def _extract_chunk_to_files(
        self,
        output_path: Path,
        start_time: float,
        end_time: float,
        output_format: str,
        pix_fmt: str,
        preserve_hdr: bool,
        tonemap_method: Optional[str],
        frame_offset: int
    ) -> int:
        """Extract frames from a specific time chunk to files."""
        stream = ffmpeg.input(str(self.input_file), ss=start_time)

        chunk_duration = end_time - start_time
        stream = stream.filter('trim', duration=chunk_duration)
        stream = stream.filter('setpts', 'PTS-STARTPTS')

        stream = self._apply_color_processing(stream, preserve_hdr, tonemap_method)

        output_pattern = str(output_path / f'frame_%06d.{output_format}')

        # Use start_number to continue frame numbering from previous chunks
        stream = ffmpeg.output(
            stream,
            output_pattern,
            pix_fmt=pix_fmt,
            start_number=frame_offset,
            **{'qscale:v': 1}
        )

        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

        # Count extracted frames
        extracted_frames = len(list(output_path.glob(f'frame_*.{output_format}')))
        return extracted_frames - frame_offset

    def extract_frames_to_files(
        self,
        output_dir: str,
        output_format: str = 'png',
        pix_fmt: str = 'rgb48be',
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None,
        chunk_size_gb: float = 10.0
    ) -> None:
        """
        Extract all frames from video and save to individual files.

        This is the main entry point for frame extraction. For large files,
        processing is automatically chunked to manage memory efficiently.

        Args:
            output_dir: Directory to save extracted frames
            output_format: Output format ('png', 'tiff', 'exr')
            pix_fmt: Pixel format for output (rgb48be for 16-bit HDR)
            preserve_hdr: If True, preserve HDR metadata and color space
            tonemap_method: Tone mapping algorithm if converting to SDR
                           ('hable', 'reinhard', 'mobius')
            chunk_size_gb: Internal chunk size for large files (default 10 GB)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_size_gb = self._file_size_bytes / (1024 ** 3)

        # For small files, extract directly without chunking
        if file_size_gb <= chunk_size_gb:
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
            return

        # For large files, process in chunks automatically
        chunks = self._get_chunk_boundaries(chunk_size_gb)
        logger.info(f"Large file detected ({file_size_gb:.2f} GB). "
                   f"Processing in {len(chunks)} chunks...")

        total_frames = 0
        for chunk in chunks:
            chunk_idx = chunk['chunk_index']
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: "
                       f"{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")

            frames_in_chunk = self._extract_chunk_to_files(
                output_path=output_path,
                start_time=chunk['start_time'],
                end_time=chunk['end_time'],
                output_format=output_format,
                pix_fmt=pix_fmt,
                preserve_hdr=preserve_hdr,
                tonemap_method=tonemap_method,
                frame_offset=total_frames
            )

            total_frames += frames_in_chunk
            logger.info(f"Chunk {chunk_idx + 1} complete. "
                       f"Extracted {frames_in_chunk} frames (total: {total_frames})")

        logger.info(f"Frame extraction complete. Total frames: {total_frames}")
