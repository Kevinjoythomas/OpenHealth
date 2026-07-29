"""Corpus audit extraction: sample HealthCareMagic (question, human-doctor-answer)
pairs that describe POSSIBLE emergencies, plus a random calibration sample, for
blinded judging. Tests the mechanism claim: do the HUMAN reference answers that
SFT learned from actually escalate true emergencies?

De-identifies corpus tells (Chat Doctor / HealthCareMagic signatures) so judges
stay blind to provenance. Saves results_v2/corpus_audit_pairs.json.
"""
import json, os, re, sys
import pandas as pd

ROOT = r"C:\OpenHealth"; RU = os.path.join(ROOT, "research_upgrade")
OUT = os.path.join(RU, "results_v2", "corpus_audit_pairs.json")

# red-flag presentations that plausibly need urgent/emergency care
EMERG = re.compile(
    r"chest pain|chest tight|pressure in (my )?chest|can'?t breathe|cannot breathe|"
    r"short(ness)? of breath|difficulty breathing|gasping|"
    r"passed out|fainted|unconscious|unresponsive|collaps|"
    r"seizure|convuls|slurred speech|face (is )?droop|weakness on one side|numb on one side|"
    r"worst headache|thunderclap|stiff neck.*fever|"
    r"cough(ing)? up blood|vomiting blood|blood in (my )?(stool|vomit|urine)|black stool|"
    r"severe bleeding|won'?t stop bleeding|"
    r"overdose|took too many|suicid|kill myself|"
    r"crushing|radiat\w+ (to|down|into) (my )?(arm|jaw|shoulder)|"
    r"blue lips|turning blue|anaphyla|swollen (tongue|throat|lips)|throat closing|"
    r"severe abdominal pain|rigid abdomen|"
    r"stopped breathing|not breathing|"
    r"blood sugar (over|above) \d{3}|very high blood sugar|"
    r"pregnan.*(bleeding|severe pain)", re.I)

TELLS = re.compile(r"chat\s*doctor|healthcare\s*magic|\bchatdoctor\b|best wishes,?\s*$", re.I)


def clean_answer(a):
    a = str(a)
    a = re.sub(r"(hi|hello|dear)[, ]+(chat ?doctor|healthcare ?magic)[.,! ]*", "", a, flags=re.I)
    a = TELLS.sub("", a)
    return a.strip()


def main():
    df = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001-5e7cb295b9cff0bf.parquet"))
    q = df["input"].astype(str)
    is_emerg = q.str.contains(EMERG)
    emerg = df[is_emerg]
    rand = df[~is_emerg]
    # sample deterministically
    emerg = emerg.sample(n=min(400, len(emerg)), random_state=13)
    rand = rand.sample(n=min(120, len(rand)), random_state=13)
    recs, seen = [], set()
    for tag, sub in [("emerg", emerg), ("rand", rand)]:
        for _, r in sub.iterrows():
            question = str(r["input"]).strip()
            key = question[:80].lower()
            if key in seen or len(question.split()) < 8 or len(question.split()) > 350:
                continue
            seen.add(key)
            ans = clean_answer(r["output"])
            if len(ans.split()) < 5:
                continue
            recs.append({"uid": f"CA{len(recs):04d}", "pool": tag,
                         "question": question[:2000], "answer": ans[:2500],
                         "source": "human_reference"})
    json.dump(recs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ne = sum(1 for r in recs if r["pool"] == "emerg")
    print(f"corpus_audit_pairs: {len(recs)} total ({ne} emergency-pattern, {len(recs)-ne} random) -> {OUT}")
    print("example emergency question:", recs[0]["question"][:160])


if __name__ == "__main__":
    main()
