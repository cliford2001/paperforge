#!/bin/bash
#SBATCH --job-name=pforge_4vlm_modes
#SBATCH --partition=debug
#SBATCH --account=psbl_cenia__m8zvy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace1/marodrigovergaraarroyo/logs/pforge_4vlm_modes_%j.out
#SBATCH --error=/workspace1/marodrigovergaraarroyo/logs/pforge_4vlm_modes_%j.err

set -euo pipefail

WORK=/workspace1/marodrigovergaraarroyo
PYTHON=$WORK/vllm-env/bin/python
PARQUET=$WORK/sample5.parquet
RUN_ROOT=$WORK/results_4vlm_all_modes
EXTRACT_BASE=$RUN_ROOT/extract_base
LOG_DIR=$WORK/logs
PORT=8097

mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo " PaperForge 4 VLM all context modes"
echo " job=$SLURM_JOB_ID"
echo " models=qwen25_7b,minicpmv26,internvl25_8b,internvl35_38b"
echo " modes=none,bm25,full,layered"
echo " started=$(ts)"
echo " gpu=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

echo "[$(ts)] Cleaning previous run root: $RUN_ROOT"
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

cd "$WORK/paperforge"

echo "[$(ts)] Step 1: download PDFs + Docling extraction once"
"$PYTHON" v2/main.py \
    --input-parquet "$PARQUET" \
    --out-dir "$EXTRACT_BASE" \
    --keep-pdf \
    --reuse-existing-pdf

echo "[$(ts)] Extraction done: $EXTRACT_BASE"
for pmcid in "$EXTRACT_BASE"/PMC*/; do
    [ -d "$pmcid" ] || continue
    figs="$pmcid/figures.json"
    tabs="$pmcid/tables.json"
    nfigs=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('items',[])))" "$figs" 2>/dev/null || echo "?")
    ntabs=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('items',[])))" "$tabs" 2>/dev/null || echo "?")
    echo "  $(basename "$pmcid") figures=$nfigs tables=$ntabs"
done

VLLM_PID=""

cleanup() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(ts)] Stopping vLLM PID=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_vllm() {
    local model_key=$1
    local model_dir=$2
    local max_len=$3
    local tp=$4
    local extra_args=${5:-}
    local log=$LOG_DIR/vllm_${model_key}_${SLURM_JOB_ID}.log

    echo "[$(ts)] START vLLM model=$model_key dir=$model_dir max_len=$max_len tp=$tp"
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model_dir" \
        --port "$PORT" \
        --tensor-parallel-size "$tp" \
        --max-model-len "$max_len" \
        --gpu-memory-utilization 0.88 \
        $extra_args \
        > "$log" 2>&1 &
    VLLM_PID=$!
    echo "[$(ts)] vLLM PID=$VLLM_PID log=$log"

    local ready=0
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[$(ts)] vLLM ready after $((i * 5))s"
            ready=1
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$(ts)] ERROR: vLLM died for $model_key. Last 160 log lines:"
            tail -160 "$log" || true
            exit 1
        fi
        sleep 5
    done

    if [ "$ready" -eq 0 ]; then
        echo "[$(ts)] ERROR: vLLM timeout for $model_key. Last 160 log lines:"
        tail -160 "$log" || true
        exit 1
    fi
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(ts)] STOP vLLM PID=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    sleep 8
}

run_mode() {
    local model_key=$1
    local mode=$2
    local out_dir=$RUN_ROOT/${model_key}_${mode}
    shift 2

    echo "[$(ts)] RUN model=$model_key mode=$mode out=$out_dir"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    rsync -a "$EXTRACT_BASE/" "$out_dir/"

    cd "$WORK/paperforge"
    "$PYTHON" v2/main.py \
        --input-parquet "$PARQUET" \
        --out-dir "$out_dir" \
        --keep-pdf \
        --reuse-existing-pdf \
        --run-analysis \
        --server "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --context-mode "$mode" \
        "$@" \
        --max-tokens 2200 \
        --temperature 0.0 \
        --timeout 900

    echo "[$(ts)] DONE model=$model_key mode=$mode"
}

run_all_modes_for_model() {
    local model_key=$1
    run_mode "$model_key" none
    run_mode "$model_key" bm25 --top-k 6 --chunk-words 250 --chunk-overlap 40
    run_mode "$model_key" full --max-context-words 3000
    run_mode "$model_key" layered --top-k 6 --chunk-words 250 --chunk-overlap 40
}

run_model() {
    local model_key=$1
    local model_dir=$2
    local max_len=$3
    local tp=$4
    local extra_args=${5:-}

    echo ""
    echo "████ model=$model_key ████"
    start_vllm "$model_key" "$model_dir" "$max_len" "$tp" "$extra_args"
    run_all_modes_for_model "$model_key"
    stop_vllm
}

run_model "qwen25_7b" "$WORK/models/Qwen2.5-VL-7B-Instruct" 16384 1 ""
run_model "minicpmv26" "$WORK/models/MiniCPM-V-2_6" 16384 1 "--trust-remote-code"
run_model "internvl25_8b" "$WORK/models/InternVL2_5-8B" 16384 1 "--trust-remote-code"
run_model "internvl35_38b" "$WORK/models/InternVL3_5-38B" 16384 2 "--trust-remote-code"

echo "[$(ts)] Summary"
for d in "$RUN_ROOT"/*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -name 'analyses_rag.json' 2>/dev/null | wc -l)
    echo "  $(basename "$d") analyses=$n"
done

echo "============================================================"
echo " FINISHED $(ts)"
echo " results=$RUN_ROOT"
echo "============================================================"
