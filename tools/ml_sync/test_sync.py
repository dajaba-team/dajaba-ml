"""Boundary tests for external mutations; never read ML data."""

import copy
import tempfile
import unittest
from pathlib import Path

import sync


CONFIG = sync.load(Path(__file__).with_name("config.json"))


def fixture():
    rows, issues = [], []
    for index, item in enumerate(CONFIG["items"]):
        cells = [{} for _ in range(8)]
        cells[1] = {"userEnteredValue": {"stringValue": item["wbs_id"]}}
        cells[3] = {"userEnteredValue": {"stringValue": "준영"}}
        cells[5] = {"userEnteredValue": {"numberValue": 46325}}
        cells[7] = {"userEnteredValue": {"numberValue": 0}}
        rows.append({"row": index + 15, "cells": cells})
        issues.append({"key": item["jira_key"], "fields": {
            "project": {"key": "SCRUM"}, "labels": [item["marker"]],
            "assignee": {"accountId": CONFIG["jira_account_id"]}, "duedate": "2026-10-30",
            "status": {"name": "진행 중", "statusCategory": {"key": "indeterminate"}}}})
    return {"repository": CONFIG["repository"], "rows": rows, "issues": issues, "prs": []}


class FakeApi:
    def __init__(self, snapshot):
        self.data = copy.deepcopy(snapshot)
        self.writes = []

    def snapshot(self):
        return copy.deepcopy(self.data)

    def write(self, op):
        self.writes.append(op)
        if op["kind"] == "wbs_note":
            row = next(r for r in self.data["rows"] if r["row"] == op["row"])
            row["cells"][7]["note"] = op["after"]
        else:
            issue = next(i for i in self.data["issues"] if i["key"] == op["key"])
            issue["fields"]["duedate"] = op["after"]


class SyncTests(unittest.TestCase):
    def test_second_run_is_noop_and_percent_unchanged(self):
        api = FakeApi(fixture())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            plan = sync.build_plan(CONFIG, api.snapshot())
            result = sync.apply_plan(CONFIG, plan, api, Path(folder) / "journal.json")
        self.assertEqual(len(result), 5)
        self.assertEqual(sync.build_plan(CONFIG, api.snapshot())["operations"], [])
        self.assertTrue(all(sync.value(r["cells"][7]) == {"numberValue": 0} for r in api.data["rows"]))

    def test_other_or_shared_owner_blocks_everything(self):
        for owner in ["진희", "준영·진희", "전체", ""]:
            s = fixture()
            s["rows"][1]["cells"][3]["userEnteredValue"]["stringValue"] = owner
            with self.assertRaises(sync.Conflict):
                sync.build_plan(CONFIG, s)

    def test_jira_owner_change_blocks_everything(self):
        s = fixture()
        s["issues"][0]["fields"]["assignee"] = None
        with self.assertRaises(sync.Conflict):
            sync.build_plan(CONFIG, s)

    def test_duplicate_wbs_and_marker_block(self):
        for target in ["rows", "issues"]:
            s = fixture()
            s[target].append(copy.deepcopy(s[target][0]))
            with self.assertRaises(sync.Conflict):
                sync.build_plan(CONFIG, s)

    def test_row_relocation_uses_stable_id(self):
        s = fixture()
        s["rows"][0]["row"] = 79
        self.assertEqual(sync.build_plan(CONFIG, s)["operations"][0]["row"], 79)

    def test_preserves_human_notes_and_rejects_malformed_block(self):
        first = sync.merged_note("담당자 메모", "자동 영역")
        self.assertEqual(sync.merged_note(first, "자동 영역"), first)
        self.assertTrue(sync.merged_note(first, "새 영역").startswith("담당자 메모\n\n"))
        with self.assertRaises(sync.Conflict):
            sync.merged_note("--- DAJABA ML SYNC BEGIN ---", "x")

    def test_due_direction_and_formula_guard(self):
        s = fixture()
        s["issues"][0]["fields"]["duedate"] = None
        op = sync.build_plan(CONFIG, s)["operations"][0]
        self.assertEqual((op["kind"], op["after"]), ("jira_due", "2026-10-30"))
        s["rows"][0]["cells"][5]["userEnteredValue"] = {"formulaValue": "=TODAY()"}
        with self.assertRaises(sync.Conflict):
            sync.build_plan(CONFIG, s)

    def test_tampered_and_stale_plan_perform_zero_writes(self):
        for mutate in ["plan", "data"]:
            api = FakeApi(fixture())
            plan = sync.build_plan(CONFIG, api.snapshot())
            if mutate == "plan":
                plan["operations"][0]["row"] = 12
            else:
                api.data["rows"][0]["cells"][7]["note"] = "다른 사용자의 새 메모"
            with self.assertRaises(sync.Conflict):
                sync.apply_plan(CONFIG, plan, api, "unused-journal.json")
            self.assertEqual(api.writes, [])

    def test_merged_pr_never_completes_work(self):
        s = fixture()
        s["prs"] = [{"number": 1, "title": "SCRUM-33", "body": "", "state": "closed",
                     "merged_at": "2026-10-28", "html_url": "https://github.com/dajaba-team/dajaba-ml/pull/1"}]
        plan = sync.build_plan(CONFIG, s)
        self.assertTrue(all(op["kind"] == "wbs_note" for op in plan["operations"]))
        self.assertIn("PR #1", plan["operations"][0]["after"])

    def test_changed_scope_rejected(self):
        c = copy.deepcopy(CONFIG)
        c["repository"] = "dajaba-team/dajaba-ai"
        with self.assertRaises(sync.Conflict):
            sync.build_plan(c, fixture())

    def test_done_percent_mismatch_reported_without_status_write(self):
        s = fixture()
        s["issues"][0]["fields"]["status"] = {"name": "완료", "statusCategory": {"key": "done"}}
        plan = sync.build_plan(CONFIG, s)
        self.assertEqual(len(plan["warnings"]), 1)
        self.assertTrue(all(op["kind"] == "wbs_note" for op in plan["operations"]))

    def test_failed_readback_keeps_journal(self):
        class NoWriteApi(FakeApi):
            def write(self, op):
                self.writes.append(op)
        api = NoWriteApi(fixture())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            journal = Path(folder) / "journal.json"
            with self.assertRaises(sync.Conflict):
                sync.apply_plan(CONFIG, sync.build_plan(CONFIG, api.snapshot()), api, journal)
            self.assertEqual(sync.load(journal)[0]["status"], "write_returned")
            self.assertEqual(len(api.writes), 1)


if __name__ == "__main__":
    unittest.main()
