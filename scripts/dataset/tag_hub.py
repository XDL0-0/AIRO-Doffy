"""Create an explicit tag on a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a tag on a Hugging Face dataset repository."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repository, e.g. owner/name.")
    parser.add_argument("--tag", required=True, help="Tag to create, e.g. v3.0.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional source revision; the Hub default is used when omitted.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the remote write. Without this flag the command is a dry run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.execute:
        logger.info(
            "Dry run: would create tag %s on dataset %s%s",
            args.tag,
            args.repo_id,
            f" from revision {args.revision}" if args.revision else "",
        )
        logger.info("Re-run with --execute to perform the remote write.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_tag(
        args.repo_id,
        tag=args.tag,
        revision=args.revision,
        repo_type="dataset",
    )
    logger.info("Created tag %s on %s.", args.tag, args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
