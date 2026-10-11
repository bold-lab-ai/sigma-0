#!/bin/bash
# Load the canonical inference setup functions, but replace cleanup with an
# unmount-only implementation. Workspace policy forbids automated deletion.

: "${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"

source "$BENCHMARK_ROOT/pipeline/_squashfs_helpers.sh"

_infer_unmount_only() {
    local mount_root="$1"
    local d
    local -a mount_dirs=()

    if [ -z "$mount_root" ] || [ ! -d "$mount_root" ]; then
        return 0
    fi

    shopt -s nullglob
    mount_dirs=( "$mount_root"/*/ )
    shopt -u nullglob
    for d in "${mount_dirs[@]}"; do
        if mountpoint -q "$d" 2>/dev/null; then
            fusermount -u "$d" 2>/dev/null || true
        fi
    done
}

infer_squashfs_cleanup() {
    _infer_unmount_only "$INFER_SQUASHFS_MOUNT_ROOT"
    _infer_unmount_only "$INFER_WIDE_BOOK_MOUNT_ROOT"
    echo "[infer-squashfs] unmounted; preserving unique temporary roots per no-delete policy"
    echo "[infer-squashfs] mount_root=${INFER_SQUASHFS_MOUNT_ROOT:-n/a}"
    echo "[infer-squashfs] farm_root=${INFER_SQUASHFS_FARM_ROOT:-n/a}"
    echo "[infer-squashfs] wide_mount_root=${INFER_WIDE_BOOK_MOUNT_ROOT:-n/a}"
    echo "[infer-squashfs] wide_farm_root=${INFER_WIDE_BOOK_FARM_ROOT:-n/a}"
}
