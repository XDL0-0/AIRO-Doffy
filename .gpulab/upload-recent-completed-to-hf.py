#!/usr/bin/env python3
"""Upload completed recent training artifacts from shared GPULab storage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


WRM_ROOT = Path(
    "/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness"
)
ICRA_ROOT = Path(
    "/project_ghent/beaver_policies/WRM_grasp_cylinder_different_sizes_lero_tightness/"
    "icra_policy_matrix_v1_20260831"
)


def validate_wrm(folder: Path, pattern: str) -> None:
    milestones = sorted(folder.glob(pattern))
    required = [
        folder / "last.pt",
        folder / "resolved_config.yaml",
        folder / "metrics.jsonl",
    ]
    if len(milestones) != 10 or not all(path.is_file() for path in required):
        raise RuntimeError(
            f"Incomplete run {folder}: milestones={len(milestones)}, "
            f"required={[path.is_file() for path in required]}"
        )


def upload_run(
    api: HfApi,
    *,
    repo_id: str,
    folder: Path,
    path_in_repo: str,
    message: str,
) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        ignore_patterns=["*.tmp", "wandb/**", "**/wandb/**"],
        commit_message=message,
    )
    print(f"UPLOAD_COMPLETE repo={repo_id} path={path_in_repo}", flush=True)


def completed_icra_runs() -> list[Path]:
    completed: list[Path] = []
    for last in sorted(ICRA_ROOT.glob("*/seed_*/last.pt")):
        folder = last.parent
        milestones = sorted(folder.glob("*_step_*.pt"))
        required = [
            folder / "resolved_config.yaml",
            folder / "metrics.jsonl",
            folder / "train.log",
        ]
        if len(milestones) == 2 and all(path.is_file() for path in required):
            completed.append(folder)
        else:
            print(
                f"SKIP_INCOMPLETE folder={folder} milestones={len(milestones)}",
                flush=True,
            )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    if not token:
        raise SystemExit("Hugging Face token file is empty")
    os.environ["HF_TOKEN"] = token
    api = HfApi(token=token)
    identity = api.whoami()
    if identity.get("name") != "IXDLI":
        raise SystemExit(f"Expected Hugging Face user IXDLI, got {identity.get('name')!r}")
    print("HF_AUTH_OK user=IXDLI", flush=True)

    closure = WRM_ROOT / "dp_beaver_closure_cluster6_bs32/dp_beaver_closure"
    validate_wrm(closure, "dp_beaver_closure_step_*.pt")
    upload_run(
        api,
        repo_id="IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-closure",
        folder=closure,
        path_in_repo="checkpoints",
        message="Upload complete dp_beaver_closure checkpoints through step 100000",
    )

    wrap = WRM_ROOT / "wrm_wrap_stratified_cluster9_bs32/WRM_wrap"
    validate_wrm(wrap, "WRM_wrap_step_*.pt")
    upload_run(
        api,
        repo_id="IXDLI/AIRO-Doffy-WRM-Grasp-WRM-wrap",
        folder=wrap,
        path_in_repo="checkpoints",
        message="Upload complete WRM_wrap checkpoints through step 100000",
    )

    icra_repo = "IXDLI/AIRO-Doffy-ICRA-Policy-Matrix-v1"
    api.create_repo(repo_id=icra_repo, repo_type="model", private=False, exist_ok=True)
    manifest = ICRA_ROOT / "resolved_manifest.yaml"
    if not manifest.is_file():
        raise RuntimeError(f"Missing ICRA manifest: {manifest}")
    api.upload_file(
        repo_id=icra_repo,
        repo_type="model",
        path_or_fileobj=str(manifest),
        path_in_repo="resolved_manifest.yaml",
        commit_message="Upload resolved ICRA matrix manifest",
    )
    completed = completed_icra_runs()
    print(f"ICRA_COMPLETED_SNAPSHOT count={len(completed)}", flush=True)
    for folder in completed:
        relative = folder.relative_to(ICRA_ROOT).as_posix()
        upload_run(
            api,
            repo_id=icra_repo,
            folder=folder,
            path_in_repo=relative,
            message=f"Upload completed ICRA run {relative}",
        )
    print(f"ALL_UPLOADS_COMPLETE icra_runs={len(completed)}", flush=True)


if __name__ == "__main__":
    main()
