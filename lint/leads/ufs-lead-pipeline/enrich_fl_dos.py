"""
GPT enrichment for FL DOS leads (1,525 unique companies).
  - Checkpoints to fl_dos_enriched.csv so it can be resumed if interrupted
  - MS Platform lookup on found emails
  - Prints a review summary every 50 leads
  - NO Supabase / HubSpot push yet — review first, push separately

Run:
    python enrich_fl_dos.py
"""

import csv, sys, os, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "zillow-daily-scraper"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "zillow-daily-scraper" / ".env")

import requests
from utils import search_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

INPUT_CSV      = r"C:\ufs\tasks\leads\ufs_leads_clean.csv"
CHECKPOINT_CSV = r"C:\ufs\tasks\leads\fl_dos_enriched.csv"

MS_API_BASE = os.environ["MS_API_BASE_URL"]
MS_API_KEY  = os.environ["MS_API_KEY"]

CHECKPOINT_FIELDS = [
    "first_name", "last_name", "company_name", "city", "state",
    "category", "role", "email_found", "email_method",
    "ms_registered", "ms_client_type", "ms_work_orders",
]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    """Return set of company_name keys already processed."""
    done = set()
    if not Path(CHECKPOINT_CSV).exists():
        return done
    for row in csv.DictReader(open(CHECKPOINT_CSV, encoding="utf-8")):
        done.add(row["company_name"].lower().strip())
    return done


def append_checkpoint(row: dict) -> None:
    path = Path(CHECKPOINT_CSV)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECKPOINT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── MS Platform lookup ────────────────────────────────────────────────────────

def ms_lookup(emails: list[str]) -> dict:
    if not emails:
        return {}
    headers = {"Authorization": f"Bearer {MS_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(
            f"{MS_API_BASE}/api/agents/lookup",
            json={"emails": emails},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", {}).get("agents", {})
    except Exception as e:
        log.warning(f"MS lookup error: {e}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def print_batch_summary(batch: list[dict], batch_num: int) -> None:
    found = [r for r in batch if r["email_found"]]
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num} REVIEW — {len(found)}/{len(batch)} emails found")
    print(f"{'='*60}")
    for r in found:
        ms = "ALREADY ON MS" if r["ms_registered"] else "new lead"
        print(f"  [{r['category']}] {r['first_name']} {r['last_name']}")
        print(f"    company : {r['company_name']}")
        print(f"    email   : {r['email_found']}  ({r['email_method']})")
        print(f"    ms      : {ms}")
        print()
    print(f"  Not found: {len(batch) - len(found)}")
    print(f"{'='*60}\n")


def main():
    leads = list(csv.DictReader(open(INPUT_CSV, encoding="utf-8")))
    done  = load_checkpoint()

    remaining = [r for r in leads if r["company_name"].lower().strip() not in done]
    log.info(f"Total: {len(leads):,}  |  Done: {len(done):,}  |  Remaining: {len(remaining):,}")

    enriched = found = 0
    current_batch: list[dict] = []
    batch_num = len(done) // 50 + 1

    for i, lead in enumerate(remaining, 1):
        first   = lead["first_name"]
        last    = lead["last_name"]
        company = lead["company_name"]
        cat     = lead["category"]

        log.info(f"[{len(done)+i}/{len(leads)}] {first} {last} | {company} | {cat}")

        result = search_email(f"{first} {last}", company)
        emails = result.get("emails") or []
        email  = emails[0] if emails else None
        method = result.get("method", "not_found")

        log.info(f"  email: {email or 'none'}  method: {method}")
        enriched += 1

        row = {
            "first_name":     first,
            "last_name":      last,
            "company_name":   company,
            "city":           lead["city"],
            "state":          lead["state"],
            "category":       cat,
            "role":           lead["role"],
            "email_found":    email or "",
            "email_method":   method,
            "ms_registered":  "",
            "ms_client_type": "",
            "ms_work_orders": "",
        }

        if email:
            found += 1
            ms_data = ms_lookup([email])
            agent = ms_data.get(email)
            if agent:
                row["ms_registered"]   = str(agent.get("registered", ""))
                row["ms_client_type"]  = agent.get("user_type", "")
                row["ms_work_orders"]  = str(agent.get("work_order_count", 0))

        append_checkpoint(row)
        current_batch.append(row)
        time.sleep(1)

        # Print review summary every 50
        if len(current_batch) == 50:
            print_batch_summary(current_batch, batch_num)
            current_batch = []
            batch_num += 1

    # Final partial batch
    if current_batch:
        print_batch_summary(current_batch, batch_num)

    log.info(f"\n{'='*55}")
    log.info(f"Done. {enriched} processed, {found} emails found ({found/max(enriched,1)*100:.1f}%)")
    log.info(f"Results: {CHECKPOINT_CSV}")


if __name__ == "__main__":
    main()
