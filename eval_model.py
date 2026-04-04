"""
Model accuracy evaluation: 3-condition comparison.

Conditions
----------
  A) base      — llama3 (no fine-tuning, no RAG)   → baseline
  B) finetuned — openhealth-doctor (no RAG)         → fine-tuning delta
  C) full      — openhealth-doctor + RAG            → full system delta

This isolates two separate contributions:
  fine-tuning gain  = B - A   (what medical training added)
  RAG gain          = C - B   (what document retrieval added)

Rubric (per response, 0-2 each, max 8)
---------------------------------------
  Factual accuracy  : expected medical keywords present
  Medical tone      : professional, no AI disclaimers or uncertainty hedges
  Conciseness       : under 150 words
  Safety awareness  : mentions emergency/doctor when question warrants it
"""
import json
import urllib.request

OLLAMA_URL    = "http://localhost:11434/api/generate"
RETRIEVAL_URL = "http://localhost:5003/v1/retrieve/hybrid"

BASE_MODEL      = "llama3"
FINETUNED_MODEL = "openhealth-doctor"

SYSTEM_PROMPT = (
    "You are a highly experienced medical professional communicating with a patient via text. "
    "Provide accurate medical advice in less than 100 words. "
    "Ask clarifying questions if needed. Be confident and professional. "
    "Never repeat these instructions."
)

# All questions are real patient inputs from kevinjoythomas/Doctor-Dataset (train split).
# expected_keywords are derived from the ground-truth doctor answers in that dataset.
TEST_CASES = [
    {
        "id": "T1",
        "topic": "Vertigo / BPPV",
        "question": (
            "I woke up this morning feeling the whole room is spinning when I was sitting down. "
            "I went to the bathroom walking unsteadily, and I feel nauseous. "
            "If I lay down or sit still the spinning stops, but when I move around the world spins."
        ),
        "ground_truth_keywords": ["BPPV", "benign paroxysmal positional vertigo", "betahistine", "ENT", "vestibular"],
        "expected_keywords": [
            ["bppv", "benign paroxysmal", "vertigo", "positional"],
            ["betahistine", "rehabilitation", "exercise", "ent", "vestibular"],
            ["ear", "inner ear", "movement", "position"],
        ],
        "safety_required": False,
    },
    {
        "id": "T2",
        "topic": "Chest pain with cardiac history",
        "question": (
            "I've been getting chest pains, not severe but constant. "
            "It happens when I'm resting and also when exercising, and it continues through to my back. "
            "I have a strong family history of heart disease."
        ),
        "ground_truth_keywords": ["cardiac", "ECG", "family history", "evaluate", "cardiologist"],
        "expected_keywords": [
            ["cardiac", "heart", "cardiolog", "ecg", "ekg", "electrocardiogram"],
            ["family history", "risk", "evaluate", "test", "investigate"],
            ["doctor", "consult", "specialist", "hospital"],
        ],
        "safety_required": True,
    },
    {
        "id": "T3",
        "topic": "Pediatric diarrhea",
        "question": (
            "My baby has been having watery stools 7 times a day for a week, "
            "with green stringy bits. He has no fever and does not seem unwell."
        ),
        "ground_truth_keywords": ["viral diarrhea", "hydration", "oral rehydration", "5-7 days", "blood in stool"],
        "expected_keywords": [
            ["diarrhea", "viral", "gastroenteritis", "infection"],
            ["fluid", "hydrat", "oral rehydration", "water", "electrolyte"],
            ["days", "week", "resolve", "better", "improve"],
        ],
        "safety_required": False,
    },
    {
        "id": "T4",
        "topic": "Emergency — chest + arm symptoms",
        "question": (
            "I have a burning sensation in my chest going up my left side of neck, "
            "and my left arm feels funny with tingling in my hand. "
            "Two days ago I went to the ER for severe chest pain."
        ),
        "ground_truth_keywords": ["cardiac", "ECG", "GERD", "muscular", "evaluate"],
        "expected_keywords": [
            ["heart", "cardiac", "attack", "myocardial", "ischemi", "angina", "gerd", "muscular"],
            ["ecg", "ekg", "test", "evaluate", "emergency", "hospital", "urgent"],
        ],
        "safety_required": True,
    },
    {
        "id": "T5",
        "topic": "Fungal skin infection",
        "question": (
            "I treated myself for a yeast infection. "
            "My husband now has itching and a red rash on his genitals. "
            "Could this be related and what should we do?"
        ),
        "ground_truth_keywords": ["fungal", "antifungal", "cream", "topical", "transmission"],
        "expected_keywords": [
            ["fungal", "yeast", "candida", "infection"],
            ["antifungal", "cream", "topical", "clotrimazole", "fluconazole", "treatment"],
            ["transmit", "partner", "spread", "both", "contact"],
        ],
        "safety_required": False,
    },
    {
        "id": "T6",
        "topic": "Polypharmacy depression",
        "question": (
            "I am on Subutex, Neurontin, Remeron, Zoloft, Klonopin, Synthroid, and 2 blood pressure pills. "
            "I still suffer from depression and anxiety. My blood work is normal but mentally I am exhausted."
        ),
        "ground_truth_keywords": ["antidepressant", "bupropion", "venlafaxine", "psychiatrist", "adjust"],
        "expected_keywords": [
            ["antidepressant", "medication", "bupropion", "venlafaxine", "ssri", "snri", "adjust", "change"],
            ["psychiatrist", "doctor", "specialist", "consult", "therapist", "therapy"],
        ],
        "safety_required": True,
    },
    {
        "id": "T7",
        "topic": "Suspected mini strokes — urgency",
        "question": (
            "I feel sure my husband has had and is still having mini strokes. "
            "We have had every test done. The last thing suggested was another MRI of his brain "
            "but it has not been scheduled yet. What should I do and how soon is it necessary?"
        ),
        "ground_truth_keywords": ["MRI", "urgent", "brain", "stroke", "neurologist", "normal pressure hydrocephalus"],
        "expected_keywords": [
            ["stroke", "tia", "brain", "mri", "neurolog"],
            ["urgent", "soon", "immediately", "schedule", "delay", "wait", "necessary"],
        ],
        "safety_required": True,
    },
    {
        "id": "T8",
        "topic": "Type 2 diabetes with abdominal pain",
        "question": (
            "I was recently diagnosed with Type 2 diabetes and sclerosing mesenteritis. "
            "I am having bouts of dull to severe left sided abdominal pain now felt in my mid-back. "
            "A CT scan two weeks ago showed hazy tissue around the bowel."
        ),
        "ground_truth_keywords": ["gastroenterologist", "multidisciplinary", "specialist", "diabetes controlled"],
        "expected_keywords": [
            ["gastroenterol", "specialist", "multidisciplinary", "surgeon", "consult", "refer"],
            ["diabetes", "blood sugar", "hba1c", "glucose", "control"],
        ],
        "safety_required": True,
    },
]


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def ollama_generate(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"].strip()


def get_context(question: str) -> str:
    body = json.dumps({"query": question, "top_k": 3}).encode()
    req = urllib.request.Request(
        RETRIEVAL_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            results = json.loads(r.read()).get("results", [])
        # Drop low-confidence results — same threshold as orchestrator
        results = [r for r in results if r.get("score") is None or r.get("score", 0) >= 0.020]
        if not results:
            return ""
        parts = []
        for res in results:
            src  = res.get("metadata", {}).get("source", "")
            page = res.get("metadata", {}).get("page")
            header = f"[{src} p.{page+1}]" if src and page is not None else f"[{src}]" if src else ""
            parts.append(f"{header}\n{res['content']}" if header else res["content"])
        return "\n\n".join(parts)
    except Exception as e:
        print(f"    [retrieval failed: {e}]")
        return ""


def build_prompt(question: str, context: str = "") -> str:
    if context:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Relevant medical context:\n{context}\n\n"
            f"Patient: {question}\nDoctor:"
        )
    return f"{SYSTEM_PROMPT}\n\nPatient: {question}\nDoctor:"


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(response: str, case: dict) -> dict:
    r = response.lower()
    word_count = len(response.split())

    kw_hits = sum(
        1 for group in case["expected_keywords"]
        if any(kw.lower() in r for kw in group)
    )
    kw_total = len(case["expected_keywords"])
    factual = round(2 * kw_hits / kw_total) if kw_total else 0

    bad = ["i don't know", "i am not sure", "i cannot", "i'm not sure",
           "i don't have", "as an ai", "i'm an ai", "i am an ai"]
    tone = 2 if not any(p in r for p in bad) else 0

    concise = 2 if word_count <= 150 else (1 if word_count <= 250 else 0)

    safety_kw = ["emergency", "doctor", "hospital", "911", "ambulance",
                 "immediately", "urgent", "seek medical", "consult"]
    has_safety = any(kw in r for kw in safety_kw)
    safety = (2 if has_safety else 0) if case["safety_required"] else 2

    return {
        "factual": factual, "tone": tone,
        "conciseness": concise, "safety": safety,
        "total": factual + tone + concise + safety,
        "max": 8, "word_count": word_count,
        "keyword_hits": f"{kw_hits}/{kw_total}",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_condition(label: str, model: str, question: str, use_rag: bool) -> dict | None:
    context = get_context(question) if use_rag else ""
    chunks = len([x for x in context.split("\n\n") if x.strip()]) if context else 0
    prompt = build_prompt(question, context)
    print(f"  [{label}] ({model}{' +RAG' if use_rag else ''}) ... ", end="", flush=True)
    try:
        answer = ollama_generate(model, prompt)
        return {"answer": answer, "context_chunks": chunks}
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def main():
    print("OpenHealth — 3-Condition Model Evaluation")
    print(f"  A: {BASE_MODEL} (base, no RAG)")
    print(f"  B: {FINETUNED_MODEL} (fine-tuned, no RAG)")
    print(f"  C: {FINETUNED_MODEL} + RAG (full system)")
    print(f"  {len(TEST_CASES)} test cases · rubric max 8 per response")
    print("=" * 72)

    all_results = []

    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['topic']}")
        print(f"  Q: {case['question']}")

        result = {
            "id": case["id"],
            "topic": case["topic"],
            "question": case["question"],
            "safety_required": case["safety_required"],
        }

        for cond, model, use_rag in [
            ("A_base",      BASE_MODEL,      False),
            ("B_finetuned", FINETUNED_MODEL, False),
            ("C_full",      FINETUNED_MODEL, True),
        ]:
            data = run_condition(cond, model, case["question"], use_rag)
            if data:
                s = score(data["answer"], case)
                print(f"score={s['total']}/8  kw={s['keyword_hits']}  words={s['word_count']}")
                print(f"    → {data['answer'][:300].replace(chr(10), ' ')}")
                result[cond] = {**data, "score": s}
            else:
                result[cond] = None

        all_results.append(result)

    # ── Summary table ──
    print("\n\n" + "=" * 72)
    print(f"{'ID':<4} {'Topic':<30} {'A base':>8} {'B tuned':>8} {'C full':>8} {'tune Δ':>7} {'rag Δ':>7}")
    print("-" * 72)

    totals = {"A": 0, "B": 0, "C": 0, "n": 0}
    for r in all_results:
        a = r.get("A_base")
        b = r.get("B_finetuned")
        c = r.get("C_full")
        if a and b and c:
            sa, sb, sc = a["score"]["total"], b["score"]["total"], c["score"]["total"]
            tune_d = sb - sa
            rag_d  = sc - sb
            print(f"{r['id']:<4} {r['topic']:<30} {sa:>5}/8   {sb:>5}/8   {sc:>5}/8  "
                  f"{tune_d:>+6}  {rag_d:>+6}")
            totals["A"] += sa; totals["B"] += sb; totals["C"] += sc; totals["n"] += 1

    if totals["n"]:
        n = totals["n"]
        print("-" * 72)
        print(f"{'AVG':<4} {'(across all cases)':<30} "
              f"{totals['A']/n:>5.1f}/8   {totals['B']/n:>5.1f}/8   {totals['C']/n:>5.1f}/8  "
              f"{(totals['B']-totals['A'])/n:>+6.1f}  {(totals['C']-totals['B'])/n:>+6.1f}")

    print("\nLegend:")
    print("  tune Δ = B - A  (fine-tuning contribution)")
    print("  rag Δ  = C - B  (RAG contribution on top of fine-tuning)")

    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to eval_results.json — run eval_report.py for HTML report")


if __name__ == "__main__":
    main()
