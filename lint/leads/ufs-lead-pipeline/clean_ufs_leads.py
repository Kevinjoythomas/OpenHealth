"""
Clean ufs_leads_final.csv:
  1. Normalize company names (suffixes, punctuation, truncation)
  2. Remove HOA boards (keep only HOA management companies)
  3. Deduplicate to one lead per unique company
  4. Output: ufs_leads_clean.csv + summary
"""

import csv, re
from collections import defaultdict

INPUT_CSV  = r"C:\ufs\tasks\leads\ufs_leads_final.csv"
OUTPUT_CSV = r"C:\ufs\tasks\leads\ufs_leads_clean.csv"

# HOA management companies have these words — boards don't
HOA_MGMT_RE = re.compile(
    r'management|mgmt|services|solutions|professionals|consulting|advisors|group',
    re.IGNORECASE,
)

# Legal suffix normalization map
_SUFFIX_MAP = [
    (re.compile(r'\bL\.?L\.?C\.?,?\s*$',       re.I), 'LLC'),
    (re.compile(r'\bL\.?L\.?P\.?,?\s*$',       re.I), 'LLP'),
    (re.compile(r'\bInc\.?,?\s*$',             re.I), 'Inc'),
    (re.compile(r'\bIncorporated\.?,?\s*$',    re.I), 'Inc'),
    (re.compile(r'\bCorp\.?,?\s*$',            re.I), 'Corp'),
    (re.compile(r'\bCorporation\.?,?\s*$',     re.I), 'Corp'),
    (re.compile(r'\bL\.?P\.?,?\s*$',           re.I), 'LP'),
    (re.compile(r'\bP\.?A\.?,?\s*$',           re.I), 'PA'),
    (re.compile(r'\bP\.?L\.?L\.?C\.?,?\s*$',  re.I), 'PLLC'),
    (re.compile(r'\bLtd\.?,?\s*$',             re.I), 'Ltd'),
]

# Roles to prefer when picking one person per company
PREFERRED_ROLES = re.compile(
    r'president|ceo|owner|principal|director|manager|partner|vice.?president|vp',
    re.IGNORECASE,
)


def normalize_company(name: str) -> str:
    # Strip leading junk characters
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name.strip())
    # Collapse internal whitespace
    name = re.sub(r'\s+', ' ', name)
    # Remove trailing punctuation
    name = name.rstrip('.,;:- ')
    # Normalize legal suffixes
    for pattern, replacement in _SUFFIX_MAP:
        name = pattern.sub(replacement, name).strip()
    return name.strip()


def norm_key(name: str) -> str:
    """Lowercase + strip all punctuation/spaces for dedup key."""
    return re.sub(r'[^a-z0-9]', '', normalize_company(name).lower())


def is_truncated(name: str) -> bool:
    """Names from FL DOS fixed-width fields truncate at 42 chars."""
    return len(name.strip()) >= 40 and not name.strip()[-1] in '.,")'


def pick_best_lead(leads: list[dict]) -> dict:
    """Pick the most useful lead for a company (prefer named decision-maker role)."""
    for lead in leads:
        if PREFERRED_ROLES.search(lead['role']):
            return lead
    return leads[0]


def main():
    rows = list(csv.DictReader(open(INPUT_CSV, encoding='utf-8')))
    print(f"Input: {len(rows):,} rows")

    # Step 1: filter out HOA boards
    kept, dropped_boards = [], 0
    for r in rows:
        if r['category'] == 'HOA' and not HOA_MGMT_RE.search(r['company_name']):
            dropped_boards += 1
            continue
        kept.append(r)
    print(f"Dropped HOA boards: {dropped_boards:,}")

    # Step 2: normalize company names
    for r in kept:
        r['company_name'] = normalize_company(r['company_name'])

    # Step 3: group by normalized key, resolve truncation
    # Build a map: norm_key -> list of company name variants
    key_to_names: dict[str, set] = defaultdict(set)
    for r in kept:
        key_to_names[norm_key(r['company_name'])].add(r['company_name'])

    # For truncated names: if a key is a prefix of another key, merge them
    all_keys = sorted(key_to_names.keys(), key=len)
    merged: dict[str, str] = {}  # truncated_key -> canonical_key
    for i, short_key in enumerate(all_keys):
        for long_key in all_keys[i+1:]:
            if long_key.startswith(short_key) and len(long_key) - len(short_key) <= 8:
                merged[short_key] = long_key
                break

    def canonical_key(name: str) -> str:
        k = norm_key(name)
        return merged.get(k, k)

    # Step 4: group leads by canonical key, pick best per company
    groups: dict[str, list] = defaultdict(list)
    for r in kept:
        groups[canonical_key(r['company_name'])].append(r)

    # Pick best lead per company; prefer longest company name (most complete)
    final = []
    for key, leads in groups.items():
        best = pick_best_lead(leads)
        # Use the longest (least truncated) company name variant
        longest_name = max((r['company_name'] for r in leads), key=len)
        best = dict(best)
        best['company_name'] = longest_name
        final.append(best)

    # Sort: REO → HOA → PM, then company name
    cat_order = {'REO': 0, 'HOA': 1, 'Property Management': 2}
    final.sort(key=lambda x: (cat_order.get(x['category'], 9), x['company_name']))

    fieldnames = ['first_name', 'last_name', 'company_name',
                  'address', 'city', 'state', 'zip', 'category', 'role']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final)

    # Summary
    from collections import Counter
    cats = Counter(r['category'] for r in final)
    print(f"\nAfter cleanup:")
    print(f"  REO                  : {cats.get('REO', 0):,} unique companies")
    print(f"  HOA Management       : {cats.get('HOA', 0):,} unique companies")
    print(f"  Property Management  : {cats.get('Property Management', 0):,} unique companies")
    print(f"  TOTAL                : {len(final):,} unique companies")
    print(f"\nReduced from {len(rows):,} leads to {len(final):,} unique companies")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
