"""
Frame extractor for 4K HDR10 MKV files.
Extracts frames while preserving HDR metadata for post-processing.
"""

import ffmpeg
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, Iterator
import logging
import os

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
        # Try to get duration from video stream first
        if 'duration' in self.video_stream:
            return float(self.video_stream['duration'])
        # Fall back to format duration
        if 'format' in self.probe and 'duration' in self.probe['format']:
            return float(self.probe['format']['duration'])
        # Estimate from frame count and fps if available
        if 'nb_frames' in self.video_stream:
            return int(self.video_stream['nb_frames']) / self.fps
        raise ValueError("Cannot determine video duration from metadata")

    def get_file_size_bytes(self) -> int:
        """Get the video file size in bytes."""
        return self._file_size_bytes

    def get_file_size_gb(self) -> float:
        """Get the video file size in gigabytes."""
        return self._file_size_bytes / (1024 ** 3)

    def get_duration(self) -> float:
        """Get the video duration in seconds."""
        return self._duration

    def get_chunk_info(self, chunk_size_gb: float = 10.0) -> list[dict]:
        """
        Calculate chunk boundaries based on file size.

        Args:
            chunk_size_gb: Size of each chunk in gigabytes (default 10 GB)

        Returns:
            List of dicts with chunk information including start/end times
        """
        chunk_size_bytes = chunk_size_gb * (1024 ** 3)
        num_chunks = int(np.ceil(self._file_size_bytes / chunk_size_bytes))

        # Calculate time boundaries based on file size ratio
        # This assumes relatively constant bitrate
        chunks = []
        for i in range(num_chunks):
            start_ratio = (i * chunk_size_bytes) / self._file_size_bytes
            end_ratio = min(((i + 1) * chunk_size_bytes) / self._file_size_bytes, 1.0)

            start_time = start_ratio * self._duration
            end_time = end_ratio * self._duration

            chunk_info = {
                'chunk_index': i,
                'start_time': start_time,
                'end_time': end_time,
                'estimated_size_gb': (end_ratio - start_ratio) * self._file_size_bytes / (1024 ** 3),
                'duration': end_time - start_time
            }
            chunks.append(chunk_info)

        logger.info(f"Video split into {num_chunks} chunks of ~{chunk_size_gb:.2f} GB each")
        return chunks

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
        output_format: str = 'png',
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
            output_format: Output format ('png', 'tiff', 'exr')
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
        output_pattern = str(output_path / f'frame_%06d.{output_format}')
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

    def extract_frames_chunked(
        self,
        chunk_size_gb: float = 10.0,
        pix_fmt: str = 'rgb48le',
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None
    ) -> Iterator[Tuple[int, Generator[np.ndarray, None, None]]]:
        """
        Process video in chunks to handle large files (e.g., 50+ GB).

        This method divides the video into time-based chunks corresponding to
        the specified file size, yielding a generator for each chunk. This allows
        processing of very large video files without loading everything into memory.

        Args:
            chunk_size_gb: Size of each chunk in gigabytes (default 10 GB)
            pix_fmt: Pixel format ('rgb48le' for 16-bit RGB, 'rgb24' for 8-bit)
            preserve_hdr: If True, preserve HDR metadata and color space
            tonemap_method: Tone mapping algorithm if converting to SDR

        Yields:
            Tuple of (chunk_index, frame_generator) where frame_generator yields
            numpy arrays of frames for that chunk
        """
        chunks = self.get_chunk_info(chunk_size_gb)

        logger.info(f"Processing video in {len(chunks)} chunks of ~{chunk_size_gb:.2f} GB each")
        logger.info(f"Total file size: {self.get_file_size_gb():.2f} GB")

        for chunk in chunks:
            chunk_idx = chunk['chunk_index']
            start_time = chunk['start_time']
            end_time = chunk['end_time']

            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: "
                       f"{start_time:.2f}s - {end_time:.2f}s "
                       f"(~{chunk['estimated_size_gb']:.2f} GB)")

            # Create a generator for this chunk
            chunk_generator = self._extract_chunk_frames(
                start_time=start_time,
                end_time=end_time,
                pix_fmt=pix_fmt,
                preserve_hdr=preserve_hdr,
                tonemap_method=tonemap_method,
                chunk_index=chunk_idx
            )

            yield (chunk_idx, chunk_generator)

    def _extract_chunk_frames(
        self,
        start_time: float,
        end_time: float,
        pix_fmt: str,
        preserve_hdr: bool,
        tonemap_method: Optional[str],
        chunk_index: int
    ) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from a specific time range (internal method for chunking).

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            pix_fmt: Pixel format
            preserve_hdr: Whether to preserve HDR
            tonemap_method: Tone mapping method
            chunk_index: Index of the current chunk (for logging)

        Yields:
            numpy.ndarray: Frame data as numpy array
        """
        # Determine bytes per pixel based on format
        if 'rgb48' in pix_fmt or 'rgb16' in pix_fmt:
            bytes_per_pixel = 6
            dtype = np.uint16
        else:
            bytes_per_pixel = 3
            dtype = np.uint8

        # Build ffmpeg command with seek optimization
        # Using -ss before -i for fast seeking
        stream = ffmpeg.input(
            str(self.input_file),
            ss=start_time  # Seek to start position (fast seek)
        )

        # Calculate duration for this chunk
        chunk_duration = end_time - start_time

        # Add time limit using -t (duration from seek point)
        stream = stream.filter('trim', duration=chunk_duration)
        stream = stream.filter('setpts', 'PTS-STARTPTS')  # Reset timestamps

        # Apply color processing
        stream = self._apply_color_processing(stream, preserve_hdr, tonemap_method)

        # Output to pipe
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt=pix_fmt)

        process = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=True)

        frame_size = self.width * self.height * bytes_per_pixel
        frame_count = 0

        try:
            while True:
                in_bytes = process.stdout.read(frame_size)

                if not in_bytes:
                    break

                if len(in_bytes) != frame_size:
                    logger.warning(f"Chunk {chunk_index}: Incomplete frame {frame_count}, skipping")
                    break

                frame = np.frombuffer(in_bytes, dtype=dtype).reshape([self.height, self.width, 3])

                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Chunk {chunk_index}: Processed {frame_count} frames")

                yield frame

        finally:
            process.stdout.close()
            process.wait()
            logger.info(f"Chunk {chunk_index}: Total frames processed: {frame_count}")

    def extract_all_frames_chunked(
        self,
        chunk_size_gb: float = 10.0,
        pix_fmt: str = 'rgb48le',
        preserve_hdr: bool = True,
        tonemap_method: Optional[str] = None
    ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """
        Extract all frames from video in chunks, yielding each frame with metadata.

        This is a convenience method that flattens the chunk-based processing,
        yielding each frame along with its chunk index and frame index within chunk.

        Args:
            chunk_size_gb: Size of each chunk in gigabytes (default 10 GB)
            pix_fmt: Pixel format ('rgb48le' for 16-bit RGB, 'rgb24' for 8-bit)
            preserve_hdr: If True, preserve HDR metadata and color space
            tonemap_method: Tone mapping algorithm if converting to SDR

        Yields:
            Tuple of (chunk_index, frame_index_in_chunk, frame_array)
        """
        total_frames = 0

        for chunk_idx, chunk_generator in self.extract_frames_chunked(
            chunk_size_gb=chunk_size_gb,
            pix_fmt=pix_fmt,
            preserve_hdr=preserve_hdr,
            tonemap_method=tonemap_method
        ):
            frame_idx = 0
            for frame in chunk_generator:
                yield (chunk_idx, frame_idx, frame)
                frame_idx += 1
                total_frames += 1

        logger.info(f"Chunked processing complete. Total frames: {total_frames}")

    def should_use_chunking(self, threshold_gb: float = 10.0) -> bool:
        """
        Determine if chunking should be used based on file size.

        Args:
            threshold_gb: File size threshold in GB above which chunking is recommended

        Returns:
            True if file size exceeds threshold, False otherwise
        """
        return self.get_file_size_gb() > threshold_gb


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

    # Method 4: CHUNKED PROCESSING for very large files (50+ GB)
    # Processes video in user-defined chunks (e.g., 10 GB at a time)
    # Prevents memory issues when loading massive video files
    print(f"\nFile size: {extractor.get_file_size_gb():.2f} GB")
    print(f"Video duration: {extractor.get_duration():.2f} seconds")

    # Check if chunking is recommended
    if extractor.should_use_chunking(threshold_gb=10.0):
        print("Large file detected, using chunked processing...")

        # Option 4a: Process each chunk separately
        # Useful when you need to handle each chunk as a unit
        # for chunk_idx, chunk_frames in extractor.extract_frames_chunked(
        #     chunk_size_gb=10.0,  # Process 10 GB at a time
        #     pix_fmt='rgb48le',
        #     preserve_hdr=True
        # ):
        #     print(f"\n--- Processing Chunk {chunk_idx} ---")
        #     frame_count = 0
        #     for frame in chunk_frames:
        #         # Process frame
        #         frame_count += 1
        #         if frame_count >= 3:  # Process just a few frames per chunk for demo
        #             break
        #     print(f"Chunk {chunk_idx}: Processed {frame_count} frames")

        # Option 4b: Process all frames with chunk metadata
        # Simpler API - yields (chunk_idx, frame_idx_in_chunk, frame)
        for chunk_idx, frame_idx, frame in extractor.extract_all_frames_chunked(
            chunk_size_gb=10.0,  # User-defined chunk size
            pix_fmt='rgb48le',
            preserve_hdr=True
        ):
            print(f"Chunk {chunk_idx}, Frame {frame_idx}: shape={frame.shape}")

            # Break after a few frames for demonstration
            if chunk_idx >= 1 or frame_idx >= 5:
                break

    # Get chunk information without processing
    # chunks = extractor.get_chunk_info(chunk_size_gb=10.0)
    # for chunk in chunks:
    #     print(f"Chunk {chunk['chunk_index']}: "
    #           f"{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s "
    #           f"(~{chunk['estimated_size_gb']:.2f} GB)")
