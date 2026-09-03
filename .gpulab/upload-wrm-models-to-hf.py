#!/usr/bin/env python3
"""Upload the three completed WRM runs from project storage to private HF repos."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(
    "/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness"
)
SPECS = {
    "adaptive_all125": {
        "repo": "IXDLI/WRM_adaptive-all125",
        "folder": PROJECT_ROOT
        / "wrm_adaptive_all125_cluster12_bs32/WRM_adaptive_all_train",
        "variant": "WRM_adaptive",
        "job": "920957a7-7b2a-4e3f-89c6-ab0e5f151af8",
    },
    "delta_all125": {
        "repo": "IXDLI/WRM_delta-all125",
        "folder": PROJECT_ROOT
        / "wrm_delta_all125_cluster9_bs32/WRM_delta_all_train",
        "variant": "WRM_delta",
        "job": "e50cbdd4-cb5b-445b-93fc-a54de5ec3d46",
    },
    "delta_split": {
        "repo": "IXDLI/WRM_delta",
        "folder": PROJECT_ROOT / "wrm_delta_cluster9_bs32/WRM_delta",
        "variant": "WRM_delta",
        "job": "5bffdea4-5db8-4fd8-825f-37974aa66219",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", choices=tuple(SPECS))
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()

    spec = SPECS[args.spec]
    folder = Path(spec["folder"])
    required = [folder / "last.pt", folder / "resolved_config.yaml", folder / "metrics.jsonl"]
    checkpoints = sorted(folder.glob("*.pt"))
    logs = sorted(folder.glob("*train.log"))
    if (
        not all(path.is_file() for path in required)
        or len(checkpoints) < 2
        or len(logs) != 1
    ):
        raise SystemExit(f"Incomplete source folder: {folder}")

    token = args.token_file.read_text().strip()
    if not token:
        raise SystemExit("HF token file is empty")
    api = HfApi(token=token)
    identity = api.whoami()
    if identity.get("name") != "IXDLI":
        raise SystemExit(f"Expected HF user IXDLI, got {identity.get('name')!r}")

    repo_id = str(spec["repo"])
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=(
            "*.pt",
            "resolved_config.yaml",
            "metrics.jsonl",
            "*train.log",
        ),
        commit_message=f"Upload completed {spec['variant']} 100k-step training run",
    )

    checksum = sha256(folder / "last.pt")
    card = f"""---
library_name: pytorch
pipeline_tag: robotics
tags:
- robotics
- imitation-learning
- diffusion-policy
---

# {repo_id.split('/', 1)[1]}

Complete checkpoint series for `{spec['variant']}` trained on the local
`WRM_grasp_cylinder_different_sizes_lero_tightness` LeRobot dataset.

- GPULab job: `{spec['job']}`
- Training steps: through `100000` using the run's configured save interval
- Checkpoints: all numbered milestones plus final EMA-enabled `last.pt`
- SHA-256: `{checksum}`
- Source configuration: `resolved_config.yaml`
- Metrics: `metrics.jsonl`

The repository is private and intended for AIRO-Doffy real-robot evaluation.
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add checkpoint provenance and usage metadata",
    )
    files = api.list_repo_files(repo_id, repo_type="model")
    expected = {
        *(path.name for path in checkpoints),
        "resolved_config.yaml",
        "metrics.jsonl",
        logs[0].name,
        "README.md",
    }
    missing = expected.difference(files)
    if missing:
        raise SystemExit(f"HF upload verification failed for {repo_id}: {sorted(missing)}")
    print(
        f"UPLOAD_COMPLETE repo={repo_id} sha256={checksum} "
        f"checkpoints={len(checkpoints)} files={len(expected)}"
    )


if __name__ == "__main__":
    main()
