"""
Preprocessor for video files.
Converts HDR10 content to SDR intermediate or processes SDR content for frame extraction.
"""

import subprocess
from pathlib import Path
from fractions import Fraction

import ffmpeg


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

        is_hdr_transfer = color_transfer.lower() in hdr_transfers
        is_hdr_primaries = color_primaries.lower() in hdr_primaries

        # Also check for side data containing HDR metadata
        side_data_list = self.video_stream.get('side_data_list', [])
        has_hdr_metadata = any(
            'mastering_display' in str(sd).lower() or 'content_light' in str(sd).lower()
            for sd in side_data_list
        )

        return is_hdr_transfer or is_hdr_primaries or has_hdr_metadata

    def preprocess(
        self,
        output_file: str,
        tonemap_method: str = 'hable',
        peak_nits: int = 1000,
        deband: bool = False,
        num_threads: int = 0
    ) -> Path:
        """
        Convert video to intermediate file suitable for frame extraction.

        For HDR content: Converts HDR10 to SDR with tone mapping.
        For SDR content: Processes with basic filters for consistency.

        Args:
            output_file: Path for output intermediate file
            tonemap_method: Tone mapping algorithm ('hable', 'reinhard', 'mobius') - HDR only
            peak_nits: Source peak luminance in nits - HDR only
            deband: Apply debanding filter (removes color banding artifacts)
            num_threads: Number of threads (0 = auto-detect)

        Returns:
            Path to generated intermediate file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_hdr:
            filters = self._build_hdr_filter_chain(tonemap_method, peak_nits, deband)
        else:
            filters = self._build_sdr_filter_chain(deband)

        cmd = [
            'ffmpeg',
            '-i', str(self.input_file),
            '-vf', ','.join(filters),
            '-c:v', 'ffv1',  # Lossless codec
            '-pix_fmt', 'rgb24',
            '-threads', str(num_threads),
            '-y',  # Overwrite
            str(output_path)
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600, shell=False)
        return output_path

    def _build_hdr_filter_chain(
        self,
        tonemap_method: str,
        peak_nits: int,
        deband: bool
    ) -> list:
        """Build FFmpeg filter chain for HDR to SDR conversion."""

        # Validate and normalize tonemap_method
        allowed_tonemap = {"hable", "reinhard", "mobius"}
        if not isinstance(tonemap_method, str):
            raise TypeError("tonemap_method must be a string")
        tonemap_method = tonemap_method.lower().strip()
        if tonemap_method not in allowed_tonemap:
            raise ValueError(
                f"Invalid tonemap_method: {tonemap_method!r}. Allowed: {sorted(allowed_tonemap)}"
            )

        # Validate peak_nits range and type
        if not isinstance(peak_nits, int):
            raise TypeError("peak_nits must be an integer number of nits")
        if not (100 <= peak_nits <= 10000):
            raise ValueError("peak_nits must be in the range [100, 10000]")

        filters = [
            # HDR to Linear light
            f'zscale=transfer=linear:npl={peak_nits}',
            # Apply tone mapping with desaturation for out-of-gamut colors
            f'tonemap={tonemap_method}:desat=2:peak={peak_nits}',
            # Linear to BT.709 SDR
            'zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=limited',
            # Format conversion
            'format=rgb24'
        ]

        if deband:
            # Reduce color banding (common in gradients after tone mapping)
            filters.append('deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1')

        return filters

    def _build_sdr_filter_chain(self, deband: bool) -> list:
        """Build FFmpeg filter chain for SDR content processing."""
        filters = [
            # Ensure consistent color space (BT.709 SDR)
            'zscale=matrix=bt709:primaries=bt709:transfer=bt709:range=limited',
            # Format conversion to RGB24
            'format=rgb24'
        ]

        if deband:
            # Reduce color banding artifacts
            filters.append('deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1')

        return filters
