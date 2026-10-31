#!/usr/bin/env python3
"""Command-line interface for the SocioDrift Video AI Pipeline.

Supports single-video analysis and multi-video batch processing.

Examples:
    # Single-video processing
    python main.py --input samples/sample_video.mp4

    # Batch processing an entire folder
    python main.py --input-dir samples/

    # Custom segment duration and output directory
    python main.py --input samples/sample_video.mp4 --segment-duration 4.0 --output-dir custom_outputs/

    # Deterministic offline demo mode
    python main.py --input samples/sample_video.mp4 --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path so package imports work seamlessly when run directly
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sociodrift.batch_processor import BatchProcessor
from sociodrift.pipeline import VideoAIPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_single_summary(result) -> None:
    """Print a clean visual summary of single-video execution."""
    print("\n" + "=" * 60)
    print("           SOCIODRIFT VIDEO AI - EXECUTION SUMMARY           ")
    print("=" * 60)
    print(f"File            : {result.filename}")
    print(f"Status          : {result.status}")
    print(f"Duration        : {result.video_duration:.2f} seconds")
    print(f"Resolution      : {result.resolution}")
    print(f"Framerate       : {result.fps:.1f} FPS (Total frames: {result.total_frames})")
    print(f"Target Segments : {result.total_segments} (Interval: {result.segment_duration_target}s)")
    print("-" * 60)
    print("Detected Scenes & Musical Prompts:")
    for scene in result.scenes:
        print(f"\n  [{scene.scene_id.upper()}] ({scene.start_time:.1f}s -> {scene.end_time:.1f}s)")
        print(f"    Mood / Role   : {scene.mood.upper()} | Story Role: {scene.story_role}")
        print(f"    Music Style   : {scene.music_cue.get('music_style')} @ {scene.music_cue.get('tempo')} BPM (Energy: {scene.music_cue.get('energy')}%)")
        print(f"    Instrument    : {scene.music_cue.get('primary_instrument')} ({scene.music_cue.get('instrumentation')})")
        print(f"    Labels        : {', '.join(scene.semantic_labels)}")
        # Print first 2 lines of prompt as preview
        prompt_lines = scene.prompt.splitlines()
        preview = " | ".join(prompt_lines[:3])
        print(f"    Prompt Preview: {preview}...")
    print("=" * 60 + "\n")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SocioDrift Video AI Pipeline - Automated Video Segmentation & AI Music Prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to a single input video file (e.g. samples/sample_video.mp4)",
    )
    input_group.add_argument(
        "--input-dir",
        "-d",
        type=str,
        help="Path to a directory of video files for batch processing (e.g. samples/)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Directory where structured JSON outputs will be saved (default: 'outputs')",
    )
    parser.add_argument(
        "--segment-duration",
        "-s",
        type=float,
        default=3.0,
        help="Length of each segment in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in deterministic demo mode without external video decoders or APIs",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging output",
    )
    return parser.parse_args(args)


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            print(f"Error: Input video not found: {args.input}", file=sys.stderr)
            return 1

        pipeline = VideoAIPipeline(
            segment_duration=args.segment_duration,
            demo_mode=args.demo,
        )

        try:
            result = pipeline.process_video(input_path, output_dir=output_dir)
            out_json = output_dir / f"{input_path.stem}.json"
            print_single_summary(result)
            print(f"Output saved to: {out_json}")
            return 0
        except Exception as exc:
            print(f"\nExecution Failed: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    elif args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
            return 1

        processor = BatchProcessor(
            segment_duration=args.segment_duration,
            demo_mode=args.demo,
        )

        try:
            batch_result = processor.process_directory(
                input_dir=input_dir, output_dir=output_dir
            )
            print(batch_result.format_console_summary())
            summary_file = output_dir / "batch_summary.json"
            print(f"Batch summary JSON saved to: {summary_file}")
            return 0 if (batch_result.successful_count > 0 or batch_result.total_files_discovered == 0) else 1
        except Exception as exc:
            print(f"\nBatch Execution Failed: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
