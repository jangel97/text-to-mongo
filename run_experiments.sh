#!/usr/bin/env bash
# Continue on errors — each step is independent, don't lose progress
set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

FAILURES=0

run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "=== $step_name ==="
    echo "=== $(date) ==="
    echo "=========================================="
    if "$@"; then
        echo "=== PASSED: $step_name ==="
    else
        echo "=== FAILED: $step_name (exit code $?) ==="
        FAILURES=$((FAILURES + 1))
    fi
}

run_step "Verify CUDA" python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

run_step "Step 1/4: Baseline — Qwen2.5-Coder-7B" \
    python3 -m text_to_mongo.training baseline --model qwen2.5-coder-7b

run_step "Step 2/4: Train — Qwen2.5-Coder-7B r=8" \
    python3 -m text_to_mongo.training train --model qwen2.5-coder-7b --lora-r 8

run_step "Step 3/4: Train — Qwen2.5-Coder-7B r=16" \
    python3 -m text_to_mongo.training train --model qwen2.5-coder-7b --lora-r 16

# Only eval adapters that actually got saved
for run in qwen2.5-coder-7b_r8 qwen2.5-coder-7b_r16; do
    model="${run%_r*}"
    if [ -d "runs/${run}/adapter" ]; then
        run_step "Eval — ${run}" \
            python3 -m text_to_mongo.training eval --model "$model" --adapter "runs/${run}/adapter" --run-name "$run"
    else
        echo "=== SKIPPED eval for ${run}: no adapter found ==="
        FAILURES=$((FAILURES + 1))
    fi
done

run_step "Step 4/4: Comparison table" \
    python3 -m text_to_mongo.training compare

echo ""
echo "=========================================="
echo "=== FINISHED at $(date) ==="
echo "=== Failures: $FAILURES ==="
echo "=========================================="

if [ $FAILURES -gt 0 ]; then
    echo "Some steps failed. Check the log above for details."
    exit 1
fi
