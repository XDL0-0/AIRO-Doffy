#!/usr/bin/env bash

set -Eeuo pipefail

ssh_alias="gpulab-b79502fa-proxy"
remote_wrm_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness"
remote_icra_root="/project_ghent/beaver_policies/WRM_grasp_cylinder_different_sizes_lero_tightness/icra_policy_matrix_v1_20260831"
local_root="/home/yuyuan/AIRO-Doffy/policies/downloaded"
partial_root="${local_root}/.partial-recent-training"
icra_local="${local_root}/icra_policy_matrix_v1_20260831"

mkdir -p "${partial_root}" "${icra_local}"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

copy_complete_run() {
    local remote_dir="$1"
    local local_name="$2"
    local expected_pattern="$3"
    local expected_count="$4"
    local destination="${local_root}/${local_name}"
    local staging_parent="${partial_root}/${local_name}"
    local remote_basename
    remote_basename="$(basename "${remote_dir}")"

    if [[ -f "${destination}/last.pt" ]]; then
        log "SKIP ${local_name}: local last.pt already exists"
        return
    fi

    local remote_count
    remote_count="$(ssh -o BatchMode=yes "${ssh_alias}" \
        "test -f '${remote_dir}/last.pt' && find '${remote_dir}' -maxdepth 1 -type f -name '${expected_pattern}' | wc -l")"
    if [[ "${remote_count}" != "${expected_count}" ]]; then
        log "SKIP ${local_name}: remote checkpoint count ${remote_count}, expected ${expected_count}"
        return
    fi

    mkdir -p "${staging_parent}"
    log "START ${local_name} from ${remote_dir}"
    scp -rp "${ssh_alias}:${remote_dir}" "${staging_parent}/"
    if [[ ! -f "${staging_parent}/${remote_basename}/last.pt" ]]; then
        log "ERROR ${local_name}: downloaded last.pt is missing"
        return 1
    fi
    mv "${staging_parent}/${remote_basename}" "${destination}"
    rmdir "${staging_parent}" 2>/dev/null || true
    log "DONE ${local_name}: $(du -sh "${destination}" | awk '{print $1}')"
}

copy_icra_seed() {
    local policy="$1"
    local seed="$2"
    local remote_dir="${remote_icra_root}/${policy}/${seed}"
    local destination_parent="${icra_local}/${policy}"
    local destination="${destination_parent}/${seed}"
    local staging_parent="${partial_root}/icra/${policy}"

    if [[ -f "${destination}/last.pt" ]]; then
        log "SKIP ICRA ${policy}/${seed}: local last.pt already exists"
        return
    fi
    if ! ssh -o BatchMode=yes "${ssh_alias}" "test -f '${remote_dir}/last.pt'"; then
        log "SKIP ICRA ${policy}/${seed}: remote last.pt is absent"
        return
    fi

    mkdir -p "${destination_parent}" "${staging_parent}"
    log "START ICRA ${policy}/${seed}"
    scp -rp "${ssh_alias}:${remote_dir}" "${staging_parent}/"
    if [[ ! -f "${staging_parent}/${seed}/last.pt" ]]; then
        log "ERROR ICRA ${policy}/${seed}: downloaded last.pt is missing"
        return 1
    fi
    mv "${staging_parent}/${seed}" "${destination}"
    log "DONE ICRA ${policy}/${seed}: $(du -sh "${destination}" | awk '{print $1}')"
}

copy_complete_run \
    "${remote_wrm_root}/dp_beaver_closure_cluster6_bs32/dp_beaver_closure" \
    "dp_beaver_closure" \
    "dp_beaver_closure_step_*.pt" \
    "10"

copy_complete_run \
    "${remote_wrm_root}/wrm_wrap_stratified_cluster9_bs32/WRM_wrap" \
    "WRM_wrap" \
    "WRM_wrap_step_*.pt" \
    "10"

if [[ ! -f "${icra_local}/resolved_manifest.yaml" ]]; then
    scp -p "${ssh_alias}:${remote_icra_root}/resolved_manifest.yaml" "${icra_local}/"
fi

while read -r policy seed; do
    copy_icra_seed "${policy}" "${seed}"
done <<'EOF'
joint_only seed_42
joint_only seed_43
joint_only seed_44
joint_vision seed_42
joint_vision seed_43
joint_vision seed_44
joint_beaver_static_key4 seed_42
joint_beaver_static_key4 seed_43
joint_beaver_static_key4 seed_44
joint_beaver_temporal_key4 seed_42
joint_beaver_temporal_key4 seed_43
joint_beaver_temporal_key4 seed_44
EOF

log "All currently completed recent runs have been downloaded"
