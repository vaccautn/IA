"""Create an offline passage review pack without running models."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vacca_video.passage_bcs_export import export_passage_labels  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_passage_labels(args.run, args.output, source_video=args.source_video)
    except (OSError, ValueError, RecursionError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(json.dumps(dict(output=str(args.output.absolute()), rows=result["rows"], assets=result["assets"])))


if __name__ == "__main__":
    main()
