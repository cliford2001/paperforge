#!/usr/bin/env python3
"""
run_figures_pipeline.py
────────────────────────────────────────────────────────────────────────
Script standalone para correr el pipeline de figuras sin SLURM.
Todos los parámetros están hardcodeados abajo en CONFIG.

Lanzar:
    python run_figures_pipeline.py

Para forzar reprocesamiento:
    python run_figures_pipeline.py --force

Para procesar un solo PDF:
    python run_figures_pipeline.py --pdf paper_nitrate.pdf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ██  CONFIGURACIÓN — edita aquí según tu cluster  ██
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "project_dir": "/workspace1/rubenqc/.ibio/chart-to-text",
    "internvl_model": "internvl3-78b",
    "chandra_model": "chandra-ocr-2",
    "tensor_parallel_size": 4,      # ajusta según GPUs disponibles en lascar
    "gpu_memory_utilization": 0.80,
    "max_model_len": 8192,
    "dpi": 300,
    "force": True,
    "no_chandra": True,
    "allow_full_page_fallback": False,
    "env": {
        "OMP_NUM_THREADS":      "8",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "LD_LIBRARY_PATH":      "/workspace1/rubenqc/venv_vllm/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/lib/python3/dist-packages/torch/lib",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# No editar debajo de aquí
# ─────────────────────────────────────────────────────────────────────────────


def gpu_info() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                print(f"  GPU {parts[0]}: {parts[1]} | "
                      f"total={parts[2]} MiB | libre={parts[3]} MiB")
    except Exception as e:
        print(f"  [WARN] nvidia-smi no disponible: {e}")


def build_cmd(
    project_dir: Path,
    internvl_model_path: Path,
    chandra_model_path: Path,
    cfg: dict,
    pdf_path: Path | None,
    phase: str,           # "ocr" | "internvl"
) -> list[str]:
    """Construye el comando Python para cada fase."""

    pipeline_script = project_dir / "src" / "pipeline_figures_nitrogen_v3.py"

    cmd = [sys.executable, str(pipeline_script)]

    if pdf_path:
        cmd += ["--pdf", str(pdf_path)]

    cmd += ["--dpi", str(cfg["dpi"])]

    if cfg["force"]:
        cmd.append("--force")

    if phase == "ocr":
        cmd.append("--no-internvl")
        if cfg["no_chandra"]:
            cmd.append("--no-chandra")
        else:
            cmd += ["--chandra-model-path", str(chandra_model_path),
                    "--chandra-backend", "hf_lowlevel"]
        if cfg["allow_full_page_fallback"]:
            cmd.append("--allow-full-page-fallback")

    elif phase == "internvl":
        cmd += [
            "--internvl-only",
            "--internvl-mode", "offline",
            "--internvl-model-path", str(internvl_model_path),
            "--tp-size", str(cfg["tensor_parallel_size"]),
            "--gpu-util", str(cfg["gpu_memory_utilization"]),
            "--max-model-len", str(cfg["max_model_len"]),
        ]

    return cmd


def run_phase(cmd: list[str], env: dict, phase_name: str) -> int:
    """Ejecuta una fase y muestra output en tiempo real."""
    print(f"\n{'='*62}")
    print(f"  {phase_name}")
    print(f"{'='*62}")
    print(f"  CMD: {' '.join(cmd[:6])} ...")

    full_env = {**os.environ, **env}
    if "LD_LIBRARY_PATH" in env:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        full_env["LD_LIBRARY_PATH"] = f"{env['LD_LIBRARY_PATH']}:{existing}" if existing else env["LD_LIBRARY_PATH"]
    proc = subprocess.Popen(
        cmd,
        env=full_env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    proc.wait()
    return proc.returncode


def main() -> None:
    # ── CLI mínimo (solo --force y --pdf como overrides) ────────────────────
    parser = argparse.ArgumentParser(
        description="Pipeline de figuras standalone (parámetros en CONFIG)."
    )
    parser.add_argument(
        "--force", action="store_true", default=None,
        help="Override: forzar reprocesamiento aunque existan outputs."
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="Override: procesar solo este PDF (nombre o ruta completa)."
    )
    parser.add_argument(
        "--tp-size", type=int, default=None,
        help="Override: número de GPUs para tensor parallel."
    )
    parser.add_argument(
        "--gpu-util", type=float, default=None,
        help="Override: gpu_memory_utilization."
    )
    parser.add_argument(
        "--ocr-only", action="store_true",
        help="Solo fase 1 (OCR + crops), sin InternVL."
    )
    parser.add_argument(
        "--internvl-only", action="store_true",
        help="Solo fase 2 (InternVL), reusar OCR/crops previos."
    )
    args = parser.parse_args()

    # Aplicar overrides CLI sobre CONFIG
    cfg = dict(CONFIG)
    if args.force is not None:
        cfg["force"] = args.force
    if args.tp_size is not None:
        cfg["tensor_parallel_size"] = args.tp_size
    if args.gpu_util is not None:
        cfg["gpu_memory_utilization"] = args.gpu_util

    # ── Rutas ────────────────────────────────────────────────────────────────
    project_dir         = Path(cfg["project_dir"])
    internvl_model_path = project_dir / "models" / cfg["internvl_model"]
    chandra_model_path  = project_dir / "models" / cfg["chandra_model"]
    raw_pdfs_dir        = project_dir / "data" / "raw_pdfs"

    # PDF específico o None (todos)
    pdf_path: Path | None = None
    if args.pdf:
        p = Path(args.pdf)
        pdf_path = p if p.is_absolute() else raw_pdfs_dir / p
        if not pdf_path.exists():
            print(f"[ERROR] PDF no encontrado: {pdf_path}")
            sys.exit(1)

    # ── Validaciones ─────────────────────────────────────────────────────────
    if not project_dir.exists():
        print(f"[ERROR] project_dir no existe: {project_dir}")
        sys.exit(1)

    pipeline_script = project_dir / "src" / "pipeline_figures_nitrogen_v3.py"
    if not pipeline_script.exists():
        print(f"[ERROR] No se encontró el pipeline: {pipeline_script}")
        sys.exit(1)

    if not args.ocr_only and not internvl_model_path.exists():
        print(f"[ERROR] Modelo InternVL no encontrado: {internvl_model_path}")
        print(f"  Descárgalo con:")
        print(f"  hf download OpenGVLab/InternVL3-78B "
              f"--local-dir {internvl_model_path}")
        sys.exit(1)

    # ── Header ───────────────────────────────────────────────────────────────
    t_global = time.time()
    print("=" * 62)
    print("  PIPELINE FIGURAS NITRÓGENO v3.2 — STANDALONE")
    print("=" * 62)
    print(f"  project_dir  : {project_dir}")
    print(f"  internvl     : {internvl_model_path.name}")
    print(f"  tp_size      : {cfg['tensor_parallel_size']} GPU(s)")
    print(f"  gpu_util     : {cfg['gpu_memory_utilization']}")
    print(f"  max_model_len: {cfg['max_model_len']}")
    print(f"  dpi          : {cfg['dpi']}")
    print(f"  force        : {cfg['force']}")
    print(f"  pdf          : {pdf_path or '(todos en raw_pdfs/)'}")
    print()

    # PDFs a procesar
    if pdf_path:
        pdfs = [pdf_path]
    else:
        pdfs = sorted(raw_pdfs_dir.glob("*.pdf"))
        if not pdfs:
            print(f"[ERROR] No hay PDFs en {raw_pdfs_dir}")
            sys.exit(1)

    print(f"PDFs a procesar: {len(pdfs)}")
    for p in pdfs:
        print(f"  - {p.name}")

    print()
    print("Estado inicial de GPUs:")
    gpu_info()

    # ── Aplicar variables de entorno ─────────────────────────────────────────
    env = cfg["env"]

    # ── FASE 1: OCR + crops ──────────────────────────────────────────────────
    if not args.internvl_only:
        cmd_ocr = build_cmd(
            project_dir=project_dir,
            internvl_model_path=internvl_model_path,
            chandra_model_path=chandra_model_path,
            cfg=cfg,
            pdf_path=pdf_path,
            phase="ocr",
        )
        rc = run_phase(cmd_ocr, env, "FASE 1: OCR + extracción de crops")
        if rc != 0:
            print(f"\n[ERROR] Fase 1 falló con código {rc}")
            sys.exit(rc)

        # Resumen fase 1
        print("\n── Resumen fase 1 " + "─" * 44)
        captions_dir = project_dir / "data" / "figure_captions"
        from collections import Counter
        import json
        total_cands = 0
        for cand_file in sorted(captions_dir.glob("*.candidates.json")):
            try:
                data = json.loads(cand_file.read_text())
                methods = Counter(c["crop_quality"] for c in data)
                total_cands += len(data)
                m_str = "  ".join(f"{k}={v}" for k, v in sorted(methods.items()))
                stem = cand_file.name.replace(".candidates.json", "")
                print(f"  {stem:<35} {len(data):>3} candidatos  |  {m_str}")
            except Exception:
                pass
        print(f"  {'TOTAL':<35} {total_cands:>3} candidatos")

        if total_cands == 0:
            print("\n[WARN] No hay figuras candidatas. No se ejecutará InternVL.")
            sys.exit(0)

    # ── FASE 2: InternVL ──────────────────────────────────────────────────────
    if not args.ocr_only:
        print()
        print("Estado GPUs antes de InternVL:")
        gpu_info()

        cmd_internvl = build_cmd(
            project_dir=project_dir,
            internvl_model_path=internvl_model_path,
            chandra_model_path=chandra_model_path,
            cfg=cfg,
            pdf_path=pdf_path,
            phase="internvl",
        )
        rc = run_phase(cmd_internvl, env, "FASE 2: InternVL interpretación de crops")
        if rc != 0:
            print(f"\n[ERROR] Fase 2 falló con código {rc}")
            sys.exit(rc)

    # ── Resumen final ─────────────────────────────────────────────────────────
    import json
    from collections import Counter

    print("\n" + "=" * 62)
    print("  RESUMEN FINAL")
    print("=" * 62)

    final_text_dir = project_dir / "data" / "final_text"
    total_figs = total_acc = total_rej = 0
    total_low = total_med = total_high = 0

    for json_path in sorted(final_text_dir.glob("*.figures.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        figs   = data["figures"]
        levels = Counter(
            f.get("validation", {}).get("validation_level", "?") for f in figs
        )
        accepted = sum(
            1 for f in figs if f.get("validation", {}).get("accepted_for_training")
        )
        stem = json_path.name.replace(".figures.json", "")

        total_figs += len(figs)
        total_acc  += accepted
        total_rej  += len(figs) - accepted
        total_low  += levels.get("low_risk", 0)
        total_med  += levels.get("medium_risk", 0)
        total_high += levels.get("high_risk", 0)

        print(f"\n{'─'*62}")
        print(f"  Paper  : {data.get('paper_title','?')[:58]}")
        print(f"  Stem   : {stem}")
        print(f"  Figuras: {len(figs)}  aceptadas={accepted}  "
              f"rechazadas={len(figs)-accepted}")
        print(f"  Riesgo : low={levels.get('low_risk',0)}  "
              f"medium={levels.get('medium_risk',0)}  "
              f"high={levels.get('high_risk',0)}")
        print()
        print(f"  {'figure_id':<26} {'crop_quality':<22} {'type':<18} "
              f"{'unc':<5} {'hal':<5} ok  level")
        print("  " + "─" * 95)
        for f in figs:
            v  = f.get("validation")     or {}
            i  = f.get("interpretation") or {}
            ok = "✓" if v.get("accepted_for_training") else "✗"
            print(
                f"  {f['figure_id']:<26} "
                f"{f['candidate']['crop_quality']:<22} "
                f"{i.get('figure_type','?'):<18} "
                f"{i.get('uncertainty_level','?'):<5} "
                f"{i.get('hallucination_risk','?'):<5} "
                f"{ok}   "
                f"{v.get('validation_level','?')}"
            )

    if total_figs > 0:
        pct = 100 * total_acc // total_figs
        print(f"\n{'='*62}")
        print(f"  TOTAL: {total_figs} figuras  |  "
              f"aceptadas={total_acc} ({pct}%)  rechazadas={total_rej}")
        print(f"  Riesgo: low={total_low}  "
              f"medium={total_med}  high={total_high}")

    elapsed = time.time() - t_global
    print(f"\n  Tiempo total: {elapsed:.1f}s")
    print(f"  Outputs: {final_text_dir}")
    print(f"  Crops  : {project_dir / 'data' / 'extracted_figures'}")
    print()


if __name__ == "__main__":
    main()