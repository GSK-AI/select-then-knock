#!/usr/bin/env bash
# Capture the per-arm GNU-time sidecars required by the scaling memory plot.
# No scheduler, private paths, separate environments, or installation commands.
# Memory is GNU time's "Maximum resident set size": the peak RSS of the largest
# single process in the run, not the sum over worker processes.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Run from the project root:
  bash experiments/07-computational-scaling/run_resource_grid.sh

Requires an existing Python environment and GNU time (Linux /usr/bin/time,
or an already installed gtime on macOS). Does not install anything.
Environment: PYTHON, GNU_TIME, OUT_ROOT, P_GRID, SEED_GRID, ARM_GRID.
Defaults: four feature counts, ten seeds, and both scaling-study arms
(stk_secondary, direct_model_x).
Refuses existing cell directories so skipped fits cannot become timing data.
This is a full, potentially expensive grid; it does not run during extraction.
HELP
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'Unknown arguments; use --help.\n' >&2
    exit 2
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
GNU_TIME="${GNU_TIME:-/usr/bin/time}"
OUT_ROOT="${OUT_ROOT:-$HERE/outputs}"
if ! "$GNU_TIME" --version 2>/dev/null | grep -q 'GNU'; then
    printf 'GNU time is required. Select an existing GNU_TIME executable; no installation is automatic.\n' >&2
    exit 2
fi
read -r -a counts <<< "${P_GRID:-500 2000 5000 20000}"
read -r -a seeds <<< "${SEED_GRID:-42 43 44 45 46 47 48 49 50 51}"
read -r -a arms <<< "${ARM_GRID:-stk_secondary direct_model_x}"

for p in "${counts[@]}"; do
    for seed in "${seeds[@]}"; do
        out="$OUT_ROOT/p-$p/seed-$seed"
        if [[ -e "$out" ]]; then
            printf 'Refusing existing cell directory: %s. Use a fresh OUT_ROOT.\n' "$out" >&2
            exit 2
        fi
        mkdir -p "$out"
        for arm in "${arms[@]}"; do
            printf 'p=%s seed=%s arm=%s\n' "$p" "$seed" "$arm"
            "$GNU_TIME" -v -o "$out/resources-$arm.txt" env \
                STK_SOLVER_BACKEND=celer N_SAMPLES=800 P_VALUES="$p" K=3 SNR=0.5 RHO=0.8 \
                SEEDS="$seed" OUTDIR="$out" ARMS="$arm" "$PYTHON" "$HERE/run.py"
        done
    done
done
