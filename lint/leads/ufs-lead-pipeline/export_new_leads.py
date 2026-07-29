"""
Export unsent Zillow leads from Supabase as a CSV in HubSpot import format.
Marks exported rows as sent by setting sent_at = now().

Output columns (match HubSpot import template exactly):
  Detail URL, Status Text, Address, State, Zipcode,
  Broker Name, Agent Name, Listing Agent Phone, Email,
  Record Source, Website URL, Company Address, Company State, Company Zip

Usage:
  python export_new_leads.py               # export all unsent leads
  python export_new_leads.py --dry-run     # preview without marking as sent
  python export_new_leads.py --out my.csv  # custom output path
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "zillow-daily-scraper/.env"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]

COLUMNS = [
    "Detail URL", "Status Text", "Address", "State", "Zipcode",
    "Broker Name", "Agent Name", "Listing Agent Phone", "Email",
    "Record Source", "Website URL", "Company Address", "Company State", "Company Zip",
    "Agent Status",
]

COL_MAP = {
    "Detail URL":          "detail_url",
    "Status Text":         "status_text",
    "Address":             "address",
    "State":               "state",
    "Zipcode":             "zipcode",
    "Broker Name":         "broker_name",
    "Agent Name":          "agent_name",
    "Listing Agent Phone": "listing_agent_phone",
    "Email":               "email",
    "Record Source":       "record_source",
    "Website URL":         "website_url",
    "Company Address":     "company_address",
    "Company State":       "company_state",
    "Company Zip":         "company_zip",
    "Agent Status":        "ufs_agent_status",
}


def fetch_unsent(supabase) -> list[dict]:
    db_cols = "email," + ",".join(COL_MAP.values())
    resp = (
        supabase.table("leads")
        .select(db_cols)
        .is_("sent_at", "null")
        .neq("email", "")
        .execute()
    )
    return resp.data or []


def mark_sent(supabase, emails: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    # Supabase filter supports up to ~1000 values in `in_`; chunk if needed
    chunk_size = 500
    for i in range(0, len(emails), chunk_size):
        chunk = emails[i : i + chunk_size]
        supabase.table("leads").update({"sent_at": now}).in_("email", chunk).execute()


def write_csv(rows: list[dict], out_path: str) -> int:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(db_col, "") for col, db_col in COL_MAP.items()})
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Export unsent Zillow leads for HubSpot import")
    parser.add_argument("--dry-run", action="store_true", help="Export CSV but do NOT mark leads as sent")
    parser.add_argument("--out", help="Output CSV path (auto-named if omitted)")
    args = parser.parse_args()

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Fetching unsent leads …")
    rows = fetch_unsent(supabase)
    print(f"  {len(rows)} unsent leads with email found")

    if not rows:
        print("Nothing to export.")
        sys.exit(0)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = args.out or os.path.join(os.path.dirname(__file__), f"hubspot_leads_{today}.csv")
    count = write_csv(rows, out_path)
    print(f"  Wrote {count} rows -> {out_path}")

    if args.dry_run:
        print("  [dry-run] sent_at NOT updated.")
    else:
        emails = [r["email"] for r in rows if r.get("email")]
        mark_sent(supabase, emails)
        print(f"  Marked {len(emails)} rows as sent (sent_at = now).")


if __name__ == "__main__":
    main()
