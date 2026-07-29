"""
Build final UFS client leads CSV from re_leads.csv.

Filters:
  - Categories: HOA, Property Management, REO only
  - Roles: decision-makers only (drop lawyers, CPAs, registered agents)
  - Names: must have both first and last name
  - No duplicate (first+last+company) rows

Output: ufs_leads_final.csv
"""

import csv
import re
from collections import defaultdict

INPUT_CSV  = r"C:\ufs\tasks\leads\re_leads.csv"
OUTPUT_CSV = r"C:\ufs\tasks\leads\ufs_leads_final.csv"

TARGET_CATEGORIES = {"HOA", "Property Management", "REO"}

# Roles that indicate non-decision-makers — skip these
SKIP_ROLE_PATTERNS = re.compile(
    r"esq|atty|attorney|cpa|c\.p\.a|counsel|legal|notary|"
    r"\bra\b|reg.agent|registered.agent|trustee|fiduciary|"
    r"accountant|auditor|bookkeep|tax|paralegal|clerk",
    re.IGNORECASE,
)

# Company name patterns that indicate it's a law firm or accounting firm, not a property company
SKIP_COMPANY_PATTERNS = re.compile(
    r"\bp\.a\.\b|\blaw\b|\blaw firm\b|attorney|legal group|legal services|"
    r"\bcpa\b|accounting|financial group|financial services|financial advisor|"
    r"title company|title group|insurance|mortgage|lending|bank\b",
    re.IGNORECASE,
)

# Clean up middle initials from first_name field (e.g. "BRADFORD      W" -> "BRADFORD")
def clean_first(first: str) -> str:
    # Remove trailing single letter (middle initial)
    cleaned = re.sub(r"\s+[A-Z]\s*$", "", first.strip())
    # Remove Jr, Sr, II, III, IV suffixes and embedded role codes
    cleaned = re.sub(r"\s+(JR\.?|SR\.?|II|III|IV|ESQ\.?|CPA|MGR|MGRM|LMGR|AMBR|MEMB|VP|CEO|PRES|DIR)$",
                     "", cleaned.strip(), flags=re.IGNORECASE)
    # Keep only the first word if the rest looks like a role/title code
    parts = cleaned.strip().split()
    if len(parts) > 1 and re.fullmatch(r"[A-Z]{2,6}", parts[-1]):
        cleaned = " ".join(parts[:-1])
    return cleaned.strip().title()

def clean_last(last: str) -> str:
    return last.strip().title()

def clean_company(name: str) -> str:
    # Truncate at comma for "THOMAS G. SHERMAN, P.A." type names
    # But keep "ARVANA property management inc, Suite 200" type addresses out
    return name.strip().title()


def main():
    rows = list(csv.DictReader(open(INPUT_CSV, encoding="utf-8")))
    print(f"Input rows: {len(rows)}")

    kept = []
    skipped_category  = 0
    skipped_role      = 0
    skipped_company   = 0
    skipped_no_name   = 0
    skipped_duplicate = 0

    seen = set()

    for r in rows:
        # 1. Category filter
        if r["category"] not in TARGET_CATEGORIES:
            skipped_category += 1
            continue

        # 2. Skip law firms / accounting firms as the company itself
        if SKIP_COMPANY_PATTERNS.search(r["company_name"]):
            skipped_company += 1
            continue

        # 3. Skip non-decision-maker roles
        role = r["role"].strip()
        if role and SKIP_ROLE_PATTERNS.search(role):
            skipped_role += 1
            continue

        # 4. Must have a real first and last name
        first = clean_first(r["first_name"])
        last  = clean_last(r["last_name"])
        if not first or not last or len(first) < 2 or len(last) < 2:
            skipped_no_name += 1
            continue

        # 5. Skip if name looks like a company (all caps acronym or has LLC/Inc)
        if re.search(r"\b(llc|inc|corp|ltd|p\.a\.|pa)\b", first, re.IGNORECASE):
            skipped_no_name += 1
            continue

        # 6. Florida addresses only — require FL explicitly
        state = r["state"].strip().upper()
        if state != "FL":
            skipped_no_name += 1
            continue

        # 7. Skip numeric-only roles (HOA board seat numbers like "1","2","3")
        if role and re.fullmatch(r"\d+", role):
            skipped_role += 1
            continue

        # 8. Deduplicate on (first, last, company)
        key = (first.lower(), last.lower(), r["company_name"].lower())
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)

        kept.append({
            "first_name":   first,
            "last_name":    last,
            "company_name": clean_company(r["company_name"]),
            "address":      r["address"].strip().title(),
            "city":         r["city"].strip().title(),
            "state":        r["state"].strip().upper(),
            "zip":          r["zip"].strip(),
            "category":     r["category"],
            "role":         role,
        })

    # Sort: REO first, then HOA, then Property Management; then by company name
    cat_order = {"REO": 0, "HOA": 1, "Property Management": 2}
    kept.sort(key=lambda x: (cat_order.get(x["category"], 9), x["company_name"]))

    fieldnames = ["first_name", "last_name", "company_name",
                  "address", "city", "state", "zip", "category", "role"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nFilters applied:")
    print(f"  Wrong category  : {skipped_category:,}")
    print(f"  Law/acct firm   : {skipped_company:,}")
    print(f"  Non-decision role: {skipped_role:,}")
    print(f"  Missing name    : {skipped_no_name:,}")
    print(f"  Duplicate       : {skipped_duplicate:,}")
    print(f"\nFinal leads      : {len(kept):,}")

    # Breakdown by category
    from collections import Counter
    cats = Counter(r["category"] for r in kept)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count:,}")

    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
