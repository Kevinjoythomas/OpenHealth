import csv
from collections import defaultdict
from datetime import datetime

INPUT_FILE = "leads_20260227_033519.csv"

def check_duplicates(filename):
    
    records = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Total records loaded: {len(records)}")

    # ── Check by zpid ──
    zpid_seen = defaultdict(list)
    for i, r in enumerate(records):
        zpid_seen[r["zpid"]].append(i)
    zpid_dupes = {z: idxs for z, idxs in zpid_seen.items() if len(idxs) > 1}

    # ── Check by address ──
    addr_seen = defaultdict(list)
    for i, r in enumerate(records):
        addr = r["address"].strip().lower()
        if addr:
            addr_seen[addr].append(i)
    addr_dupes = {a: idxs for a, idxs in addr_seen.items() if len(idxs) > 1}

    # ── Check by agent_name ──
    agent_seen = defaultdict(list)
    for i, r in enumerate(records):
        name = r["agentName"].strip().lower()
        if name:
            agent_seen[name].append(i)
    agent_dupes = {n: idxs for n, idxs in agent_seen.items() if len(idxs) > 1}

    # ── Report ──
    print(f"\n=== DUPLICATE REPORT ===")
    print(f"Duplicate zpids:    {len(zpid_dupes)} ({sum(len(v)-1 for v in zpid_dupes.values())} extra rows)")
    print(f"Duplicate addresses:{len(addr_dupes)} ({sum(len(v)-1 for v in addr_dupes.values())} extra rows)")
    print(f"Duplicate agents:   {len(agent_dupes)} ({sum(len(v)-1 for v in agent_dupes.values())} extra rows)")

    if zpid_dupes:
        print(f"\nDuplicate zpids:")
        for zpid, idxs in list(zpid_dupes.items())[:10]:
            print(f"  zpid {zpid} — rows {idxs}")

    if agent_dupes:
        print(f"\nAgents with multiple listings (top 10):")
        for name, idxs in sorted(agent_dupes.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {name}: {len(idxs)} listings")

    # ── Deduplicate by zpid (keep first occurrence) ──
    seen_zpids = set()
    unique_records = []
    for r in records:
        zpid = r["zpid"]
        if zpid not in seen_zpids:
            seen_zpids.add(zpid)
            unique_records.append(r)

    print(f"\n=== DEDUP RESULT ===")
    print(f"Before: {len(records)}")
    print(f"After:  {len(unique_records)}")
    print(f"Removed:{len(records) - len(unique_records)}")

    # ── Save deduped CSV ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"leads_deduped_{timestamp}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=unique_records[0].keys())
        writer.writeheader()
        writer.writerows(unique_records)

    print(f"\nSaved: {out_file}")

if __name__ == "__main__":
    check_duplicates(INPUT_FILE)