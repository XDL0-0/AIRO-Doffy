#!/home/yuyuan/miniconda3/bin/python
"""Download only the selected public Hugging Face checkpoints."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


JOBS = (
    (
        "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-closure",
        Path("policies/downloaded/dp_beaver_closure"),
        ("step_040000.pt", "step_050000.pt", "last.pt"),
    ),
    (
        "IXDLI/AIRO-Doffy-WRM-Grasp-WRM-wrap",
        Path("policies/downloaded/WRM_wrap"),
        ("step_040000.pt", "step_050000.pt", "last.pt"),
    ),
    (
        "IXDLI/AIRO-Doffy-ICRA-Policy-Matrix-v1",
        Path("policies/downloaded/icra_policy_matrix_v1_20260831"),
        ("step_050000.pt", "last.pt"),
    ),
)


def wanted(repo_id: str, name: str, suffixes: tuple[str, ...]) -> bool:
    if not name.endswith(".pt"):
        return False
    if repo_id.endswith("ICRA-Policy-Matrix-v1"):
        return name.endswith("/last.pt") or name.endswith("step_050000.pt")
    return any(name.endswith(suffix) for suffix in suffixes)


api = HfApi()
expected: list[Path] = []

for repo_id, local_dir, suffixes in JOBS:
    files = sorted(
        name
        for name in api.list_repo_files(repo_id=repo_id)
        if wanted(repo_id, name, suffixes)
    )
    if not files:
        raise RuntimeError(f"No selected checkpoints found in {repo_id}")
    print(f"SELECTED {repo_id}: {len(files)} files", flush=True)
    for name in files:
        print(f"  {name}", flush=True)
        expected.append(local_dir / name)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=files,
        max_workers=4,
    )

missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise RuntimeError("Missing downloaded files: " + ", ".join(map(str, missing)))

total = sum(path.stat().st_size for path in expected)
print(f"DOWNLOAD_COMPLETE files={len(expected)} bytes={total}", flush=True)
