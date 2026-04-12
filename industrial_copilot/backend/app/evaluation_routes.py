"""
evaluation_routes.py — FastAPI endpoints for model evaluation metrics and plots.

Endpoints:
    GET  /api/evaluation/{machine_id}                    → Metrics JSON
    GET  /api/evaluation/{machine_id}/plots/{plot_name}  → Serve plot image
    POST /api/evaluation/{machine_id}/run                → Trigger fresh evaluation
    GET  /api/evaluation/summary                         → All machines summary
"""

import os
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

log = logging.getLogger(__name__)
router = APIRouter()

EVAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "evaluation"
)


def _get_machine_eval_dir(machine_id: str) -> str:
    return os.path.join(EVAL_DIR, machine_id)


def _load_metrics(machine_id: str, model_type: str = "dense") -> dict:
    """Load cached metrics JSON for a machine/model."""
    metrics_path = os.path.join(
        _get_machine_eval_dir(machine_id), f"metrics_{model_type}.json"
    )
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No evaluation found for {machine_id}/{model_type}")

    with open(metrics_path) as f:
        return json.load(f)


# ── GET /api/evaluation/summary ──────────────────────────────────────────────

@router.get("/summary")
async def get_evaluation_summary():
    """
    Returns evaluation summary for all machines.
    """
    summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return {"status": "success", "data": json.load(f)}

    # Build summary on-the-fly from individual metrics files
    if not os.path.exists(EVAL_DIR):
        raise HTTPException(
            status_code=404,
            detail="No evaluations found. Run evaluation first via POST /api/evaluation/{machine_id}/run"
        )

    summary_rows = []
    for machine_dir in os.listdir(EVAL_DIR):
        machine_path = os.path.join(EVAL_DIR, machine_dir)
        if not os.path.isdir(machine_path):
            continue

        for fname in os.listdir(machine_path):
            if fname.startswith("metrics_") and fname.endswith(".json"):
                model_type = fname.replace("metrics_", "").replace(".json", "")
                try:
                    data = _load_metrics(machine_dir, model_type)
                    m = data["metrics"]
                    summary_rows.append({
                        "machine_id":       machine_dir,
                        "model_type":       model_type,
                        "accuracy":         m.get("accuracy"),
                        "precision":        m.get("precision"),
                        "recall":           m.get("recall"),
                        "f1_score":         m.get("f1_score"),
                        "auc_roc":          m.get("auc_roc"),
                        "fpr":              m.get("false_positive_rate"),
                        "fnr":              m.get("false_negative_rate"),
                        "threshold":        m.get("threshold"),
                        "separation_ratio": m.get("separation_ratio"),
                        "evaluated_at":     data.get("evaluated_at"),
                    })
                except Exception as e:
                    log.warning(f"Error loading metrics for {machine_dir}/{model_type}: {e}")

    return {"status": "success", "data": {"summary": summary_rows}}


# ── GET /api/evaluation/{machine_id} ─────────────────────────────────────────

@router.get("/{machine_id}")
async def get_evaluation(
    machine_id: str,
    model_type: str = Query("dense", description="Model type: dense or lstm")
):
    """
    Returns evaluation metrics for a specific machine and model type.
    """
    try:
        data = _load_metrics(machine_id, model_type)

        # Convert plot filenames to API URLs
        if "plots" in data:
            data["plot_urls"] = {
                name: f"/api/evaluation/{machine_id}/plots/{filename}"
                for name, filename in data["plots"].items()
            }

        return {"status": "success", "data": data}

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for {machine_id}/{model_type}. "
                   f"Run POST /api/evaluation/{machine_id}/run first."
        )
    except Exception as e:
        log.error(f"Error loading evaluation for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/evaluation/{machine_id}/plots/{plot_name} ────────────────────────

@router.get("/{machine_id}/plots/{plot_name}")
async def get_evaluation_plot(machine_id: str, plot_name: str):
    """
    Serves a specific evaluation plot image.

    plot_name examples:
        roc_curve_dense.png
        mse_distribution_dense.png
        confusion_matrix_dense.png
        threshold_sweep_dense.png
    """
    plot_path = os.path.join(_get_machine_eval_dir(machine_id), plot_name)

    if not os.path.exists(plot_path):
        raise HTTPException(
            status_code=404,
            detail=f"Plot not found: {plot_name} for {machine_id}"
        )

    return FileResponse(
        plot_path,
        media_type="image/png",
        filename=plot_name
    )


# ── POST /api/evaluation/{machine_id}/run ─────────────────────────────────────

@router.post("/{machine_id}/run")
async def run_evaluation(
    machine_id: str,
    model_type: str = Query("dense", description="Model type: dense, lstm, or both")
):
    """
    Triggers a fresh evaluation for the specified machine.
    This may take 10-30 seconds depending on dataset size.
    """
    try:
        from models.evaluate_model import evaluate_autoencoder

        results = {}
        model_types = ["dense", "lstm"] if model_type == "both" else [model_type]

        for mt in model_types:
            try:
                result = evaluate_autoencoder(machine_id, mt)
                results[mt] = result
            except FileNotFoundError as e:
                results[mt] = {"error": str(e), "status": "skipped"}
            except Exception as e:
                log.error(f"Evaluation failed for {machine_id}/{mt}: {e}")
                results[mt] = {"error": str(e), "status": "failed"}

        return {
            "status": "success",
            "message": f"Evaluation complete for {machine_id}",
            "results": results
        }

    except Exception as e:
        log.error(f"Evaluation error for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
