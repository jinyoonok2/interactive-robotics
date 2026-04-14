#!/usr/bin/env bash
#
# Run all environment, data, and system checks.
#
# Usage:
#     bash tests/check_all.sh              # Run everything
#     bash tests/check_all.sh system       # System checks only
#     bash tests/check_all.sh data         # Data/assets checks only
#     bash tests/check_all.sh habitat      # habitat-grasp env only
#     bash tests/check_all.sh uad          # uad env only
#
set -uo pipefail
# Note: not using set -e because habitat-sim has a known cleanup crash (exit 134)
# that fires after all checks pass. We track failures ourselves via EXIT_CODE.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BOLD="\033[1m"
RESET="\033[0m"
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Source conda so we can activate envs; also save path for child processes
CONDA_SH=""
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "WARNING: Could not find conda.sh — env-specific checks may fail"
fi
[[ -n "$CONDA_SH" ]] && source "$CONDA_SH"

EXIT_CODE=0
TARGET="${1:-all}"

run_check() {
    local label="$1"
    local env_name="$2"   # empty string means no env switch
    local script="$3"

    echo ""
    echo -e "${BOLD}${SEP}${RESET}"
    echo -e "${BOLD}  ${label}${RESET}"
    echo -e "${BOLD}${SEP}${RESET}"

    if [[ -n "$env_name" ]]; then
        # Spawn a completely fresh bash process to:
        # 1. Activate the conda env
        # 2. Run the check script
        # 3. Write exit code to CHECK_RESULT_FILE before Python cleanup
        # This fully isolates any SIGABRT from habitat-sim cleanup.
        local tmpf
        tmpf=$(mktemp)
        bash -c "
            source '${CONDA_SH:-/dev/null}' 2>/dev/null
            conda activate '$env_name' 2>/dev/null || exit 1
            CHECK_RESULT_FILE='$tmpf' python -u '$SCRIPT_DIR/$script' 2>/dev/null
        " 2>/dev/null || true
        local rc
        rc=$(cat "$tmpf" 2>/dev/null) || rc=1
        rm -f "$tmpf"
        if [[ "${rc:-1}" != "0" ]]; then
            EXIT_CODE=1
        fi
    else
        local tmpf
        tmpf=$(mktemp)
        CHECK_RESULT_FILE="$tmpf" python "$SCRIPT_DIR/$script" 2>&1 || true
        local rc
        rc=$(cat "$tmpf" 2>/dev/null) || rc=1
        rm -f "$tmpf"
        if [[ "${rc:-1}" != "0" ]]; then
            EXIT_CODE=1
        fi
    fi
}

# ── Dispatch ────────────────────────────────────────────────────────────────

case "$TARGET" in
    system)
        run_check "System Check" "" "check_system.py"
        ;;
    data)
        run_check "Data & Assets Check" "" "check_data.py"
        ;;
    habitat|habitat-grasp)
        run_check "habitat-grasp Environment" "habitat-grasp" "check_habitat_grasp.py"
        ;;
    uad)
        run_check "uad Environment" "uad" "check_uad.py"
        ;;
    all)
        run_check "System Check" "" "check_system.py"
        run_check "Data & Assets Check" "" "check_data.py"
        run_check "habitat-grasp Environment" "habitat-grasp" "check_habitat_grasp.py"
        run_check "uad Environment" "uad" "check_uad.py"
        ;;
    *)
        echo "Usage: $0 [system|data|habitat|uad|all]"
        exit 1
        ;;
esac

# ── Final Summary ───────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${SEP}${RESET}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${BOLD}  \033[92mAll checks passed!\033[0m${RESET}"
else
    echo -e "${BOLD}  \033[91mSome checks failed — review output above.\033[0m${RESET}"
fi
echo -e "${BOLD}${SEP}${RESET}"
echo ""

exit $EXIT_CODE
