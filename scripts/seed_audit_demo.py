"""
P-AUDIT v1 demo seed — temporarily flip ONE PendingApproval row to pending so
the Operations page renders an alert card with the new AUDIT PANEL section.

Idempotent. Re-running rebinds the same row.

Usage:
    D:/python/python.exe scripts/seed_audit_demo.py            # seed
    D:/python/python.exe scripts/seed_audit_demo.py --restore  # undo

The script prints the row id and the original status so you can manually
restore even without --restore.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory import SessionFactory, PendingApproval


SAVE_PATH = os.path.join(os.path.dirname(__file__), ".audit_demo_seed.json")


def seed() -> None:
    with SessionFactory() as s:
        if os.path.exists(SAVE_PATH):
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            row = s.get(PendingApproval, int(saved["id"]))
            if row is not None and row.status == "pending":
                print(f"  already seeded: row id={row.id} status=pending")
                return

        row = (
            s.query(PendingApproval)
             .filter(PendingApproval.status == "approved")
             .order_by(PendingApproval.id.desc())
             .first()
        )
        if row is None:
            print("  no approved row to flip; seed skipped")
            return

        saved = {
            "id":               int(row.id),
            "status":           row.status,
            "resolved_at":      row.resolved_at.isoformat() if row.resolved_at else None,
            "resolved_by":      row.resolved_by,
            "approval_deadline": row.approval_deadline.isoformat() if row.approval_deadline else None,
            "review_rationale": row.review_rationale,
            "review_category":  row.review_category,
        }
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)

        row.status = "pending"
        row.resolved_at = None
        row.resolved_by = None
        row.approval_deadline = datetime.date.today() + datetime.timedelta(days=3)
        s.commit()

        print(f"  seeded: row id={row.id} status=pending  (saved -> {SAVE_PATH})")
        print("  open Operations page to see the AUDIT PANEL section on this alert.")


def restore() -> None:
    if not os.path.exists(SAVE_PATH):
        print(f"  no saved seed at {SAVE_PATH}; nothing to restore")
        return
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        saved = json.load(f)

    with SessionFactory() as s:
        row = s.get(PendingApproval, int(saved["id"]))
        if row is None:
            print(f"  row id={saved['id']} not found; cleanup skipped")
        else:
            row.status = saved["status"]
            row.resolved_at = (
                datetime.datetime.fromisoformat(saved["resolved_at"])
                if saved["resolved_at"] else None
            )
            row.resolved_by = saved["resolved_by"]
            row.approval_deadline = (
                datetime.date.fromisoformat(saved["approval_deadline"])
                if saved["approval_deadline"] else None
            )
            row.review_rationale = saved["review_rationale"]
            row.review_category = saved["review_category"]
            s.commit()
            print(f"  restored: row id={row.id} -> status={saved['status']}")
    os.remove(SAVE_PATH)
    print(f"  removed marker {SAVE_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true",
                    help="Undo the most recent seed.")
    args = ap.parse_args()
    if args.restore:
        restore()
    else:
        seed()
