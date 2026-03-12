"""Evaluation harness for the OpenHealth Doctor model.

Measures BLEU and ROUGE-L against a small medical QA reference set,
querying the model through the live Ollama API.

Usage:
    # Against locally registered model (after setup_model.sh):
    python eval.py

    # Against the HuggingFace-pulled fallback:
    python eval.py --model hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF

    # Custom Ollama host:
    python eval.py --ollama-url http://localhost:11434

    # Save results to JSON:
    python eval.py --output eval_results.json

Requirements (install separately, not in service requirements):
    pip install evaluate rouge-score nltk requests
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference QA set — 20 medical questions with expected answer keywords.
# BLEU/ROUGE are computed against the reference strings.
# ---------------------------------------------------------------------------
REFERENCE_QA = [
    {
        "question": "What are the common symptoms of type 2 diabetes?",
        "reference": "Common symptoms of type 2 diabetes include increased thirst, frequent urination, fatigue, blurred vision, slow-healing sores, and frequent infections.",
    },
    {
        "question": "What is hypertension and what are its risk factors?",
        "reference": "Hypertension is high blood pressure. Risk factors include obesity, lack of exercise, high sodium diet, smoking, excessive alcohol, stress, age, and family history.",
    },
    {
        "question": "What are the warning signs of a heart attack?",
        "reference": "Warning signs of a heart attack include chest pain or pressure, shortness of breath, pain in arm or jaw, nausea, cold sweat, and lightheadedness.",
    },
    {
        "question": "How is pneumonia treated?",
        "reference": "Pneumonia is treated with antibiotics for bacterial pneumonia, rest, fluids, and fever-reducing medications. Severe cases may require hospitalisation.",
    },
    {
        "question": "What causes iron deficiency anaemia?",
        "reference": "Iron deficiency anaemia is caused by insufficient iron intake, poor absorption, blood loss from menstruation or gastrointestinal bleeding, or increased demand during pregnancy.",
    },
    {
        "question": "What are the symptoms of a urinary tract infection?",
        "reference": "UTI symptoms include burning urination, frequent urge to urinate, cloudy or strong-smelling urine, pelvic pain, and sometimes fever.",
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "reference": "Type 1 diabetes is an autoimmune condition where the pancreas produces no insulin. Type 2 is characterised by insulin resistance and relative insulin deficiency, usually linked to lifestyle.",
    },
    {
        "question": "How do you treat a mild fever in adults?",
        "reference": "A mild fever in adults can be managed with rest, adequate hydration, and over-the-counter antipyretics such as paracetamol or ibuprofen.",
    },
    {
        "question": "What are the symptoms of appendicitis?",
        "reference": "Appendicitis symptoms include sudden pain starting around the navel and shifting to the lower right abdomen, nausea, vomiting, loss of appetite, and low-grade fever.",
    },
    {
        "question": "What lifestyle changes help manage high cholesterol?",
        "reference": "Managing high cholesterol includes reducing saturated fat and trans fat intake, increasing exercise, eating more fibre, quitting smoking, and limiting alcohol.",
    },
    {
        "question": "What are the signs of dehydration?",
        "reference": "Signs of dehydration include dry mouth, dark urine, decreased urination, fatigue, dizziness, and confusion in severe cases.",
    },
    {
        "question": "When should a patient with chest pain seek emergency care?",
        "reference": "A patient with chest pain should seek emergency care immediately if pain is severe, radiates to the arm or jaw, is accompanied by shortness of breath, sweating, or if they have known heart disease.",
    },
    {
        "question": "What is asthma and how is it managed?",
        "reference": "Asthma is a chronic inflammatory airway disease causing wheezing, breathlessness, and chest tightness. Management includes inhaled corticosteroids, bronchodilators, avoiding triggers, and an action plan.",
    },
    {
        "question": "What are common causes of lower back pain?",
        "reference": "Common causes of lower back pain include muscle strain, herniated disc, poor posture, osteoarthritis, and in some cases kidney problems or infections.",
    },
    {
        "question": "What vaccines are recommended for adults?",
        "reference": "Recommended adult vaccines include influenza annually, tetanus-diphtheria-pertussis every 10 years, shingles for adults over 50, pneumococcal for those over 65, and COVID-19 boosters.",
    },
    {
        "question": "How is hypothyroidism diagnosed and treated?",
        "reference": "Hypothyroidism is diagnosed via TSH blood test. It is treated with daily levothyroxine (synthetic thyroid hormone) with periodic dose adjustment.",
    },
    {
        "question": "What are the symptoms of anxiety disorder?",
        "reference": "Anxiety disorder symptoms include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep disturbances.",
    },
    {
        "question": "What is the recommended treatment for a mild burn?",
        "reference": "Mild burns should be cooled under running water for 10-20 minutes, covered with a clean non-adhesive dressing. Avoid ice, butter, or breaking blisters.",
    },
    {
        "question": "What are the symptoms of a stroke?",
        "reference": "Stroke symptoms follow the FAST acronym: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Also sudden severe headache or vision loss.",
    },
    {
        "question": "How is gastroesophageal reflux disease (GERD) managed?",
        "reference": "GERD is managed with lifestyle changes (elevating head of bed, avoiding triggers), antacids, H2 blockers, or proton pump inhibitors. Severe cases may need surgery.",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OpenHealth Doctor model via Ollama")
    p.add_argument("--model", default="openhealth-doctor", help="Ollama model name")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--timeout", type=int, default=60, help="Per-request timeout seconds")
    p.add_argument("--output", default=None, help="Path to save JSON results (optional)")
    return p.parse_args()


def query_ollama(url: str, model: str, prompt: str, timeout: int) -> str:
    """Send a single prompt to Ollama and return the response text."""
    resp = requests.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU-4 and ROUGE-L over prediction/reference pairs."""
    try:
        import evaluate
    except ImportError:
        log.error("Install the 'evaluate' and 'rouge-score' packages: pip install evaluate rouge-score")
        sys.exit(1)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    rouge_score = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )
    return {
        "bleu4": round(bleu_score["bleu"], 4),
        "rouge_l": round(rouge_score["rougeL"], 4),
        "num_samples": len(predictions),
    }


def main() -> None:
    args = parse_args()

    # Verify Ollama is reachable and model exists
    try:
        r = requests.get(f"{args.ollama_url}/api/tags", timeout=10)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        if not any(args.model in m for m in available):
            log.warning(
                "Model '%s' not found in Ollama. Available: %s",
                args.model, available,
            )
            log.warning("Run `bash models/setup_model.sh` to register the local GGUF first.")
    except Exception as exc:
        log.error("Cannot reach Ollama at %s: %s", args.ollama_url, exc)
        sys.exit(1)

    log.info("Evaluating model=%s on %d QA pairs ...", args.model, len(REFERENCE_QA))

    predictions: list[str] = []
    references: list[str] = []
    per_sample: list[dict] = []

    for i, item in enumerate(REFERENCE_QA, 1):
        t0 = time.perf_counter()
        try:
            pred = query_ollama(args.ollama_url, args.model, item["question"], args.timeout)
        except Exception as exc:
            log.warning("Sample %d failed: %s", i, exc)
            pred = ""
        elapsed = (time.perf_counter() - t0) * 1000

        predictions.append(pred)
        references.append(item["reference"])
        per_sample.append({
            "question": item["question"],
            "reference": item["reference"],
            "prediction": pred,
            "latency_ms": round(elapsed, 1),
        })
        log.info("[%d/%d] latency=%.0fms", i, len(REFERENCE_QA), elapsed)

    metrics = compute_metrics(predictions, references)
    log.info("Results: BLEU-4=%.4f  ROUGE-L=%.4f  n=%d",
             metrics["bleu4"], metrics["rouge_l"], metrics["num_samples"])

    results = {
        "model": args.model,
        "metrics": metrics,
        "samples": per_sample,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        log.info("Saved to %s", args.output)
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
