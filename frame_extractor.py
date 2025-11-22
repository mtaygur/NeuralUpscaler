"""
Preprocessor for video files.
Converts HDR10 content to SDR intermediate or processes SDR content for frame extraction.
"""

import subprocess
from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass, field

import ffmpeg


@dataclass
class HdrTonemapOptions:
    """Options for HDR to SDR tone mapping."""
    method: str = 'hable'
    peak_nits: int = 1000
    desat: float = 2.0


@dataclass
class HardwareAccelOptions:
    """
    Options for hardware-accelerated decoding in FFmpeg.
    """
    enabled: bool = False
    method: str = 'auto'  # 'auto', 'cuda', 'vaapi', 'qsv', 'videotoolbox', 'dxva2', 'd3d11va'
    device: str = ''  # Device specifier (e.g., '/dev/dri/renderD128' for VAAPI, '0' for CUDA)



class Preprocessor:
    """Preprocess video files for neural network training data extraction."""

    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        self.probe = ffmpeg.probe(str(self.input_file))
        self.video_stream = next(
            (s for s in self.probe['streams'] if s['codec_type'] == 'video'),
            None
        )

        if not self.video_stream:
            raise ValueError("No video stream found in file")

        self.width = int(self.video_stream['width'])
        self.height = int(self.video_stream['height'])
        self.fps = self._parse_fps(self.video_stream)
        self.duration = self._get_duration()
        self.is_hdr = self._detect_hdr()

    @staticmethod
    def _parse_fps(stream: dict) -> float:
        """Safely parse FPS from a video stream, trying common keys."""
        for key in ("r_frame_rate", "avg_frame_rate"):
            val = stream.get(key)
            try:
                # Fraction can handle strings like "24000/1001"
                if val and val != "0/0":
                    return float(Fraction(str(val)))
            except (ValueError, ZeroDivisionError):
                continue  # Ignore malformed or zero-denominator values
        raise ValueError("Cannot determine FPS from probe data")

    def _get_duration(self) -> float:
        if 'duration' in self.video_stream:
            return float(self.video_stream['duration'])
        if 'format' in self.probe and 'duration' in self.probe['format']:
            return float(self.probe['format']['duration'])
        if 'nb_frames' in self.video_stream:
            return int(self.video_stream['nb_frames']) / self.fps
        raise ValueError("Cannot determine video duration")

    def _detect_hdr(self) -> bool:
        """Detect if the video stream contains HDR content."""
        # Check for HDR10 indicators in the video stream
        color_transfer = self.video_stream.get('color_transfer', '')
        color_primaries = self.video_stream.get('color_primaries', '')

        # HDR10 typically uses SMPTE ST 2084 (PQ) transfer and BT.2020 primaries
        hdr_transfers = {'smpte2084', 'arib-std-b67'}  # PQ and HLG
        hdr_primaries = {'bt2020'}

        is_hdr_transfer = (color_transfer or '').lower() in hdr_transfers
        is_hdr_primaries = (color_primaries or '').lower() in hdr_primaries

        # Also check for side data containing HDR metadata
        side_data_list = self.video_stream.get('side_data_list', [])
        has_hdr_metadata = any(
            'mastering display metadata' in sd.get('side_data_type', '').lower() or
            'content light level metadata' in sd.get('side_data_type', '').lower()
            for sd in side_data_list
        )

        return is_hdr_transfer or is_hdr_primaries or has_hdr_metadata

    def _detect_interlacing(self) -> bool:
        """Detect if the video stream is interlaced."""
        field_order = self.video_stream.get('field_order', 'unknown')
        # tt = top field first, bb = bottom field first, tb/bt = mixed
        return field_order in ('tt', 'bb', 'tb', 'bt')

    def _validate_interval(
        self,
        start_time: float | None,
        end_time: float | None
    ) -> tuple[float | None, float | None]:
        """
        Validate time interval bounds against video properties.

        Args:
            start_time: Start time in seconds, or None to start from beginning
            end_time: End time in seconds, or None to process until end

        Returns:
            Tuple of (start_time, end_time) in seconds, or (None, None) if both are None

        Raises:
            ValueError: If interval parameters are invalid or exceed video duration
        """

        # Validate start_time
        if start_time is not None:
            if start_time < 0:
                raise ValueError("start_time must be non-negative")
            if start_time >= self.duration:
                raise ValueError(f"start_time must be less than video duration ({self.duration:.2f}s)")

        # Validate end_time
        if end_time is not None:
            if end_time < 0:
                raise ValueError("end_time must be non-negative")
            if end_time > self.duration:
                raise ValueError(f"end_time must not exceed video duration ({self.duration:.2f}s)")

        # Validate ordering when both are specified
        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                raise ValueError("end_time must be greater than start_time")

        return start_time, end_time

    def preprocess(
        self,
        output_file: str,
        hdr_options: HdrTonemapOptions = field(default_factory=HdrTonemapOptions),
        deband: bool = False,
        hwaccel_options: HardwareAccelOptions | None = None,
        num_threads: int = 0,
        start_time: float | None = None,
        end_time: float | None = None,
        dry_run: bool = False
    ) -> Path:
        """
        Convert video to intermediate file suitable for frame extraction.

        For HDR content: Converts HDR10 to SDR with tone mapping.
        For SDR content: Processes with basic filters for consistency.

        Args:
            output_file: Path for output intermediate file
            hdr_options: Tone mapping options for HDR to SDR conversion.
            deband: Apply debanding filter.
            hwaccel_options: Hardware acceleration options for decoding. If None, software decoding is used.
            num_threads: Number of threads (0 = auto-detect)
            start_time: Optional start time in seconds for processing a subset of the video.
                       If None, starts from the beginning.
            end_time: Optional end time in seconds for processing a subset of the video.
                     If None, processes until the end.
            dry_run: If True, print the FFmpeg command without executing it.

        Returns:
            Path to generated intermediate file (or output path if dry_run=True)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate time interval bounds
        start_time, end_time = self._validate_interval(start_time, end_time)

        if self.is_hdr:
            filters = self._build_hdr_filter_chain(hdr_options, deband)
        else:
            filters = self._build_sdr_filter_chain(deband)

        # Build FFmpeg command
        cmd = ['ffmpeg']

        # Add hardware acceleration options if enabled
        if hwaccel_options and hwaccel_options.enabled:
            if hwaccel_options.method != 'auto':
                cmd.extend(['-hwaccel', hwaccel_options.method])
            else:
                cmd.extend(['-hwaccel', 'auto'])

            # Add device if specified
            if hwaccel_options.device:
                cmd.extend(['-hwaccel_device', hwaccel_options.device])

        # Add start time if specified (seek to start position)
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])

        # Add input file
        cmd.extend(['-i', str(self.input_file)])

        # Add end time if specified
        if end_time is not None:
            cmd.extend(['-to', str(end_time)])

        # Add filters and encoding options
        cmd.extend([
            '-vf', ','.join(filters),
            '-c:v', 'ffv1',  # Lossless codec
            '-pix_fmt', 'rgb24',
            '-threads', str(num_threads),
            '-y',  # Overwrite
            str(output_path)
        ])

        # Handle dry run mode or execute FFmpeg
        if dry_run:
            print("Dry run mode - FFmpeg command that would be executed:")
            print(' '.join(cmd))
        else:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600, shell=False)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}") from e

        return output_path

    def _build_filter_chain(self, core_filters: list[str], deband: bool) -> list[str]:
        """
        Build a generic FFmpeg filter chain with common operations.

        Args:
            core_filters: List of content-specific filters (HDR or SDR).
            deband: Apply debanding filter after core filters.

        Returns:
            Complete list of filter strings ready for FFmpeg.
        """
        filters = []

        # Add deinterlacing if needed (before color conversions)
        if self._detect_interlacing():
            filters.append('bwdif=mode=send_frame:parity=auto:deint=all')

        # Add core filters
        filters.extend(core_filters)

        # Add debanding if requested
        if deband:
            filters.append('deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1')

        return filters

    def _build_hdr_filter_chain(
            self,
            hdr_options: HdrTonemapOptions,
        deband: bool
    ) -> list:
        """Build FFmpeg filter chain for HDR to SDR conversion."""

        # Validate and normalize tonemap_method
        allowed_tonemap = {"hable", "reinhard", "mobius"}
        if not isinstance(hdr_options.method, str):
            raise TypeError("tonemap_method must be a string")
        tonemap_method = hdr_options.method.lower().strip()
        if tonemap_method not in allowed_tonemap:
            raise ValueError(
                f"Invalid tonemap_method: {tonemap_method!r}. Allowed: {sorted(allowed_tonemap)}"
            )

        # Validate peak_nits range and type
        if not isinstance(hdr_options.peak_nits, int):
            raise TypeError("peak_nits must be an integer number of nits")
        if not (100 <= hdr_options.peak_nits <= 10000):
            raise ValueError("peak_nits must be in the range [100, 10000]")

        # Build HDR-specific core filters
        core_filters = [
            # HDR to Linear light
            f'zscale=transfer=linear:npl={hdr_options.peak_nits}',
            # Apply tone mapping with desaturation for out-of-gamut colors
            f'tonemap={tonemap_method}:desat=2:peak={hdr_options.peak_nits}',
            # Linear to BT.709 SDR
            'zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=limited',
            # Format conversion
            'format=rgb24'
        ]

        return self._build_filter_chain(core_filters, deband)

    def _build_sdr_filter_chain(self, deband: bool) -> list:
        """Build FFmpeg filter chain for SDR content processing."""
        # Build SDR-specific core filters
        core_filters = [
            # Ensure consistent color space (BT.709 SDR)
            'zscale=matrix=bt709:primaries=bt709:transfer=bt709:range=limited',
            # Format conversion to RGB24
            'format=rgb24'
        ]

        return self._build_filter_chain(core_filters, deband)
