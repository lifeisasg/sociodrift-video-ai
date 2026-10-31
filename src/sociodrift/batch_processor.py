"""Batch processing module for multiple video inputs.

Discovers supported video files, invokes the core pipeline per input,
isolates execution failures, saves individual outputs, and outputs a summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sociodrift.pipeline import VideoAIPipeline, VideoAnalysisResult
from sociodrift.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class VideoExecutionRecord:
    """Record of execution status for a single video file in a batch."""

    filename: str
    filepath: str
    status: str  # "SUCCESS" | "FAILED" | "SKIPPED"
    duration_seconds: float | None = None
    segments_count: int = 0
    output_json: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchResult:
    """Consolidated summary of a batch run."""

    total_files_discovered: int
    total_processed: int
    successful_count: int
    failed_count: int
    skipped_count: int
    results: list[VideoExecutionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_files_discovered": self.total_files_discovered,
                "total_processed": self.total_processed,
                "successful": self.successful_count,
                "failed": self.failed_count,
                "skipped": self.skipped_count,
            },
            "records": [r.to_dict() for r in self.results],
        }

    def save_summary_json(self, output_dir: str | Path) -> Path:
        """Save batch summary JSON to output directory."""
        dir_path = Path(output_dir).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        summary_path = dir_path / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved batch summary to: %s", summary_path)
        return summary_path

    def format_console_summary(self) -> str:
        """Format a clean, readable text summary for terminal display."""
        lines = [
            "",
            "=" * 40,
            "       BATCH PROCESSING SUMMARY       ",
            "=" * 40,
            f"Total videos discovered : {self.total_files_discovered}",
            f"Successfully processed  : {self.successful_count}",
            f"Failed                  : {self.failed_count}",
            f"Skipped                 : {self.skipped_count}",
            "-" * 40,
            "File Status Details:",
        ]
        for record in self.results:
            status_tag = f"[{record.status}]"
            if record.status == "SUCCESS":
                lines.append(f"  {record.filename:<25} {status_tag} ({record.segments_count} segments)")
            else:
                reason = f" - Reason: {record.error_message}" if record.error_message else ""
                lines.append(f"  {record.filename:<25} {status_tag}{reason}")
        lines.append("=" * 40)
        return "\n".join(lines)


class BatchProcessor:
    """Manages batch video discovery, sequential execution, and fault-tolerant reporting."""

    def __init__(
        self,
        pipeline: VideoAIPipeline | None = None,
        segment_duration: float = 3.0,
        demo_mode: bool = False,
    ) -> None:
        self.pipeline = pipeline or VideoAIPipeline(
            segment_duration=segment_duration, demo_mode=demo_mode
        )

    def discover_videos(self, input_dir: str | Path) -> list[Path]:
        """Scan directory and return sorted list of supported video files.

        Args:
            input_dir: Directory path to scan.

        Returns:
            Sorted list of Path objects for supported videos.

        Raises:
            NotADirectoryError: If path does not point to a directory.
            FileNotFoundError: If input directory does not exist.
        """
        dir_path = Path(input_dir).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: {dir_path}")

        supported = VideoProcessor.SUPPORTED_EXTENSIONS
        found_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported and not f.name.startswith(".")
        ]
        return sorted(found_files, key=lambda p: p.name.lower())

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> BatchResult:
        """Execute pipeline on every video file found in the input directory.

        Args:
            input_dir: Directory containing input video files.
            output_dir: Directory where JSON outputs will be written.

        Returns:
            BatchResult object containing execution records and statistics.
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        video_files = self.discover_videos(input_dir)
        total_discovered = len(video_files)
        logger.info("Discovered %d video files in %s", total_discovered, input_dir)

        if total_discovered == 0:
            logger.warning("No supported video files discovered in %s", input_dir)
            result = BatchResult(
                total_files_discovered=0,
                total_processed=0,
                successful_count=0,
                failed_count=0,
                skipped_count=0,
                results=[],
            )
            result.save_summary_json(out_path)
            return result

        records: list[VideoExecutionRecord] = []
        success_count = 0
        failure_count = 0

        for video_path in video_files:
            logger.info("--- Batch Processing: %s ---", video_path.name)
            try:
                analysis = self.pipeline.process_video(video_path, output_dir=out_path)
                out_json = out_path / f"{video_path.stem}.json"
                records.append(
                    VideoExecutionRecord(
                        filename=video_path.name,
                        filepath=str(video_path),
                        status="SUCCESS",
                        duration_seconds=analysis.video_duration,
                        segments_count=analysis.total_segments,
                        output_json=str(out_json),
                        error_message=None,
                    )
                )
                success_count += 1
            except Exception as exc:
                logger.error("Failed processing %s: %s", video_path.name, exc)
                records.append(
                    VideoExecutionRecord(
                        filename=video_path.name,
                        filepath=str(video_path),
                        status="FAILED",
                        duration_seconds=None,
                        segments_count=0,
                        output_json=None,
                        error_message=str(exc),
                    )
                )
                failure_count += 1

        batch_result = BatchResult(
            total_files_discovered=total_discovered,
            total_processed=len(records),
            successful_count=success_count,
            failed_count=failure_count,
            skipped_count=0,
            results=records,
        )

        batch_result.save_summary_json(out_path)
        return batch_result
