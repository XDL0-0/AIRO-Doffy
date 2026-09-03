#!/usr/bin/env python3
"""Print a read-only storage inventory for one Hugging Face account."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--author", required=True)
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    api = HfApi(token=token)
    repos = [
        (model.id, "model")
        for model in api.list_models(author=args.author, full=True)
    ]
    repos.extend(
        (dataset.id, "dataset")
        for dataset in api.list_datasets(author=args.author, full=True)
    )

    def inspect(spec: tuple[str, str]) -> tuple[str, str, bool, str, str, int, int]:
        repo_id, repo_type = spec
        info = api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            files_metadata=True,
        )
        siblings = info.siblings or []
        size = sum(int(sibling.size or 0) for sibling in siblings)
        created = info.created_at.isoformat() if info.created_at else "-"
        modified = info.last_modified.isoformat() if info.last_modified else "-"
        return repo_id, repo_type, bool(info.private), created, modified, size, len(siblings)

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = sorted(pool.map(inspect, repos), key=lambda row: (row[4], row[0]))

    print("repo_id\ttype\tprivate\tcreated\tmodified\tbytes\tfiles")
    for row in rows:
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
