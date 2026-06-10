#!/bin/bash
#SBATCH --job-name=paperforge_layered
#SBATCH --partition=GPU_PARTITION
#SBATCH --account=ACCOUNT_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/paperforge_layered_%j.out
#SBATCH --error=logs/paperforge_layered_%j.err

set -euo pipefail

# Edit these paths for the target cluster.
WORK=/path/to/workdir
REPO=$WORK/paperforge/V3
PYTHON=$WORK/venv/bin/python
PARQUET=$WORK/input.parquet
MODEL_DIR=$WORK/models/Qwen2.5-VL-7B-Instruct
OUT_DIR=$WORK/results_layered
PORT=8097

mkdir -p "$WORK/logs" "$OUT_DIR"

cleanup() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$REPO"

"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port "$PORT" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.88 \
    --trust-remote-code \
    > "$WORK/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

READY=0
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        tail -120 "$WORK/logs/vllm_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    tail -120 "$WORK/logs/vllm_${SLURM_JOB_ID}.log" || true
    exit 1
fi

"$PYTHON" v2/main.py \
    --input-parquet "$PARQUET" \
    --out-dir "$OUT_DIR" \
    --keep-pdf \
    --reuse-existing-pdf \
    --run-analysis \
    --server "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --context-mode layered \
    --top-k 6 \
    --chunk-words 250 \
    --chunk-overlap 40 \
    --max-tokens 2200 \
    --temperature 0.0 \
    --timeout 900
