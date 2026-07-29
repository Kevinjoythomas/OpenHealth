"""
Extract real estate related person contacts from Florida DOS corprindata files.
Two-pass per file: find RE company IDs, then extract officer/agent names.

Output: re_leads.csv with columns:
  corp_id, role, last_name, first_name, company_name, address, city, state, zip, category
"""

import os
import csv
import re
import glob
from collections import defaultdict

INPUT_GLOB = r"C:\ufs\tasks\leads\corprindata*.txt"
OUTPUT_CSV  = r"C:\ufs\tasks\leads\re_leads.csv"

# Fixed-width column positions (0-indexed)
CORP_ID   = (0,  12)
ROLE      = (12, 16)
TYPE_COL  = 16
LAST_NAME = (17, 37)   # also company name for C rows (17-58)
FIRST_NAME= (37, 59)
ADDR1     = (59, 101)
CITY      = (101, 129)
STATE     = (129, 131)
ZIP_CODE  = (131, 141)

# Category keywords — checked in order (most specific first)
CATEGORIES = [
    ("REO",                 [r"\breo\b", r"foreclos", r"bank.?owned", r"real estate owned",
                             r"distressed asset", r"default.serv", r"asset.recovery"]),
    ("HOA",                 [r"\bhoa\b", r"homeowner", r"home.?owner", r"community.assoc",
                             r"community.mgmt", r"community.management", r"condo.?assoc",
                             r"property.owners.assoc", r"residential.assoc"]),
    ("Property Management", [r"property.mgmt", r"property.management", r"prop.mgmt",
                             r"prop.management", r"asset.mgmt", r"asset.management",
                             r"facilities.mgmt", r"facilities.management"]),
    ("Rental",              [r"rental.mgmt", r"rental.management", r"rental.serv",
                             r"leasing.mgmt", r"lease.management", r"\brental\b.*llc",
                             r"\brental\b.*inc", r"vacation.rental"]),
    ("Mortgage",            [r"\bmortgage\b", r"\blender\b", r"\blending\b",
                             r"home.loan", r"home.finance", r"mortgage.serv"]),
    ("Real Estate",         [r"real.?estate", r"\brealty\b", r"\brealtor\b",
                             r"real.?estate.?broker", r"real.?property"]),
]

_compiled = [(cat, [re.compile(p, re.IGNORECASE) for p in patterns])
             for cat, patterns in CATEGORIES]


def categorize(text: str) -> str | None:
    for cat, patterns in _compiled:
        for p in patterns:
            if p.search(text):
                return cat
    return None


def parse(line: str) -> dict:
    line = line.rstrip("\r\n")
    if len(line) < 17:
        return {}
    return {
        "corp_id":    line[CORP_ID[0]:CORP_ID[1]].strip(),
        "role":       line[ROLE[0]:ROLE[1]].strip(),
        "type":       line[TYPE_COL],
        "name_field": line[LAST_NAME[0]:59].strip(),     # full name field for C rows
        "last_name":  line[LAST_NAME[0]:LAST_NAME[1]].strip(),
        "first_name": line[FIRST_NAME[0]:FIRST_NAME[1]].strip(),
        "address":    line[ADDR1[0]:ADDR1[1]].strip()    if len(line) > ADDR1[1]  else "",
        "city":       line[CITY[0]:CITY[1]].strip()      if len(line) > CITY[1]   else "",
        "state":      line[STATE[0]:STATE[1]].strip()    if len(line) > STATE[1]  else "",
        "zip":        line[ZIP_CODE[0]:ZIP_CODE[1]].strip() if len(line) > ZIP_CODE[1] else "",
    }


def process_file(path: str, writer: csv.DictWriter) -> tuple[int, int]:
    """Two-pass: collect RE company IDs, then write person rows."""

    # Pass 1: find all real-estate company IDs in this file
    re_corps: dict[str, tuple[str, str]] = {}  # corp_id -> (category, company_name)

    with open(path, encoding="latin-1") as fh:
        for line in fh:
            row = parse(line)
            if not row or row["type"] not in ("C", "P"):
                continue
            if row["type"] == "C":
                cat = categorize(row["name_field"])
                if cat:
                    re_corps[row["corp_id"]] = (cat, row["name_field"])

    if not re_corps:
        return 0, 0

    companies_found = len(re_corps)

    # Pass 2: extract person rows for those corp IDs
    persons_written = 0
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            row = parse(line)
            if not row or row["type"] != "P":
                continue
            if row["corp_id"] not in re_corps:
                continue
            cat, company_name = re_corps[row["corp_id"]]
            writer.writerow({
                "corp_id":      row["corp_id"],
                "role":         row["role"],
                "last_name":    row["last_name"],
                "first_name":   row["first_name"],
                "company_name": company_name,
                "address":      row["address"],
                "city":         row["city"],
                "state":        row["state"],
                "zip":          row["zip"],
                "category":     cat,
            })
            persons_written += 1

    return companies_found, persons_written


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print("No corprindata files found.")
        return

    print(f"Found {len(files)} files to process")

    fieldnames = ["corp_id", "role", "last_name", "first_name",
                  "company_name", "address", "city", "state", "zip", "category"]

    total_companies = 0
    total_persons   = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(files):
            fname = os.path.basename(path)
            print(f"[{i+1}/{len(files)}] {fname} ...", end=" ", flush=True)
            companies, persons = process_file(path, writer)
            total_companies += companies
            total_persons   += persons
            print(f"{companies} companies, {persons} persons")

    print(f"\nDone. {total_companies} RE companies, {total_persons} person rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
