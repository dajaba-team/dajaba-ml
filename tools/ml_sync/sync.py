"""Scoped WBS/Jira/GitHub reconciliation. No ML imports or training commands."""

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


class Conflict(ValueError):
    """Stop before a write when scope or a precondition has changed."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config(c):
    # Deployment scope is deliberately fixed; editing JSON cannot broaden writes.
    if (c["repository"], c["jira_base"], c["jira_project"], c["sheet_id"], c["sheet_name"],
        c["spreadsheet_id"], c["jira_account_id"], c["wbs_owner"]) != (
        "dajaba-team/dajaba-ml", "https://dajaba.atlassian.net", "SCRUM", 1877998031, "WBS",
        "1ixWfhx8LldEE01EbN-13jftxnaIy71rfbyCL2tEntTg",
        "712020:3f88ab19-c990-485b-9f68-0cc5079f4735", "준영"
    ):
        raise Conflict("Deployment scope changed")
    expected = {"1.4": "SCRUM-33", "2.6": "SCRUM-34", "2.13": "SCRUM-35",
                "2.21": "SCRUM-36", "3.6": "SCRUM-37"}
    if len(c["items"]) != 5 or {i["wbs_id"]: i["jira_key"] for i in c["items"]} != expected:
        raise Conflict("Mapping changed; explicit migration required")
    for i in c["items"]:
        if i["marker"] != "dajaba-ml-wbs-" + i["wbs_id"].replace(".", "-"):
            raise Conflict("Invalid stable marker")


def sheet_rows(api_sheet):
    rows = []
    for sheet in api_sheet.get("sheets", []):
        if sheet["properties"]["sheetId"] != 1877998031:
            continue
        for grid in sheet.get("data", []):
            start = grid.get("startRow", 0)
            col = grid.get("startColumn", 0)
            for offset, row in enumerate(grid.get("rowData", [])):
                cells = [{}] * col + row.get("values", [])
                cells += [{}] * max(0, 8 - len(cells))
                rows.append({"row": start + offset + 1, "cells": cells})
    return rows


def value(cell):
    return cell.get("userEnteredValue", {})


def text_value(cell):
    v = value(cell)
    return str(v.get("stringValue", cell.get("formattedValue", v.get("numberValue", ""))))


def note_for(item, issue, prs):
    # Stable content (no run timestamps), so a second run is a no-op.
    lines = [f"[dajaba-ml-sync:v1:{item['wbs_id']}]",
             f"Jira: https://dajaba.atlassian.net/browse/{item['jira_key']}",
             f"업무 상태: {issue['fields']['status']['name']}",
             "저장소: https://github.com/dajaba-team/dajaba-ml",
             "계획: WBS / 업무 상태: Jira / 구현·리뷰 근거: GitHub",
             "완료율 자동 추정 없음. 문서·구현·검증·학습·서비스 연동을 구분."]
    if item["depends_on"]:
        lines.append("선행 ML 작업: " + ", ".join(item["depends_on"]))
    for pr in sorted(prs, key=lambda p: p["number"]):
        haystack = pr.get("title", "") + "\n" + (pr.get("body") or "")
        if re.search(r"(?<![A-Z0-9-])" + re.escape(item["jira_key"]) + r"(?![A-Z0-9-])", haystack):
            lines.append(f"PR #{pr['number']} ({pr['state']}; 병합={bool(pr.get('merged_at'))}): {pr['html_url']}")
    lines.append("PR 병합은 학습·성능·연동 완료의 단독 근거가 아님.")
    return "\n".join(lines)


def merged_note(old, managed):
    # Preserve human notes. Replace only one well-delimited managed block.
    begin, end = "--- DAJABA ML SYNC BEGIN ---", "--- DAJABA ML SYNC END ---"
    block = f"{begin}\n{managed}\n{end}"
    if begin not in old and end not in old:
        return (old.rstrip() + "\n\n" if old else "") + block
    if old.count(begin) != 1 or old.count(end) != 1 or old.index(begin) > old.index(end):
        raise Conflict("Malformed/duplicate managed note")
    start, stop = old.index(begin), old.index(end) + len(end)
    return old[:start] + block + old[stop:]


def build_plan(c, snapshot):
    validate_config(c)
    if snapshot["repository"] != c["repository"]:
        raise Conflict("Wrong repository snapshot")
    ops, warnings = [], []
    rows = snapshot["rows"]
    issues = snapshot["issues"]
    for item in c["items"]:
        matches = [r for r in rows if text_value(r["cells"][1]) == item["wbs_id"]]
        if len(matches) != 1:
            raise Conflict(f"Missing/duplicate WBS ID {item['wbs_id']}")
        row = matches[0]
        if text_value(row["cells"][3]) != c["wbs_owner"]:
            raise Conflict(f"Not solely owned: {item['wbs_id']}")
        hits = [i for i in issues if item["marker"] in i["fields"].get("labels", [])]
        if len(hits) != 1 or hits[0]["key"] != item["jira_key"]:
            raise Conflict(f"Missing/duplicate Jira mapping {item['marker']}")
        issue = hits[0]
        f = issue["fields"]
        if (f.get("assignee") or {}).get("accountId") != c["jira_account_id"]:
            raise Conflict(f"Jira ownership changed: {issue['key']}")
        if f["project"]["key"] != c["jira_project"]:
            raise Conflict("Wrong Jira project")
        if "formulaValue" in value(row["cells"][5]):
            raise Conflict("WBS deadline is a formula")
        serial = value(row["cells"][5]).get("numberValue")
        if not isinstance(serial, (int, float)) or isinstance(serial, bool):
            raise Conflict("WBS deadline must be a native date serial")
        due = (dt.date(1899, 12, 30) + dt.timedelta(days=serial)).isoformat()
        if not "2026-10-01" <= due <= "2026-11-25":
            raise Conflict("WBS deadline outside project dates")
        if f.get("duedate") != due:
            ops.append({"kind": "jira_due", "key": issue["key"], "before": f.get("duedate"),
                        "after": due})
        old = row["cells"][7].get("note", "")
        new = merged_note(old, note_for(item, issue, snapshot.get("prs", [])))
        if old != new:
            ops.append({"kind": "wbs_note", "row": row["row"], "wbs_id": item["wbs_id"],
                        "before": old, "after": new})
        pct = value(row["cells"][7]).get("numberValue")
        done = f["status"]["statusCategory"]["key"] == "done"
        if done != (pct == 1):
            warnings.append(f"{issue['key']}: Jira 완료와 WBS 100% 불일치; 검증 근거를 대조한 후 수동 판정")
    return {"schema_version": 1, "config_hash": digest(c), "snapshot_hash": digest(snapshot),
            "operations": ops, "warnings": warnings}


class Api:
    def __init__(self, config):
        self.c = config

    def request(self, service, path, method="GET", body=None):
        if service == "jira":
            base = self.c["jira_base"]
            raw = os.environ["JIRA_EMAIL"] + ":" + os.environ["JIRA_API_TOKEN"]
            auth = "Basic " + base64.b64encode(raw.encode()).decode()
        elif service == "sheets":
            base = "https://sheets.googleapis.com/v4/spreadsheets/" + self.c["spreadsheet_id"]
            auth = "Bearer " + os.environ["GOOGLE_ACCESS_TOKEN"]
        else:
            base = "https://api.github.com/repos/" + self.c["repository"]
            auth = "Bearer " + os.environ["GITHUB_TOKEN"]
        req = urllib.request.Request(base + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": auth, "Accept": "application/json",
                                              "Content-Type": "application/json", "User-Agent": "dajaba-ml-sync/1"})
        # Never echo request headers, API response bodies or tokens on failure.
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise Conflict(f"{service} HTTP {exc.code}; operation not blindly retried") from None

    def snapshot(self):
        fields = "sheets(properties(sheetId),data(startRow,startColumn,rowData(values(userEnteredValue,formattedValue,note))))"
        query = urllib.parse.urlencode({"ranges": "'WBS'!B11:H80", "includeGridData": "true", "fields": fields})
        rows = sheet_rows(self.request("sheets", "?" + query))
        issues, token = [], None
        while True:
            q = {"jql": 'project = SCRUM AND labels = "dajaba-ml-sync"',
                 "fields": "summary,assignee,status,duedate,labels,project,updated", "maxResults": 100}
            if token:
                q["nextPageToken"] = token
            page = self.request("jira", "/rest/api/3/search/jql?" + urllib.parse.urlencode(q))
            issues.extend(page["issues"])
            if page.get("isLast", not page.get("nextPageToken")):
                break
            token = page["nextPageToken"]
        prs = []
        for page in range(1, 101):
            batch = self.request("github", f"/pulls?state=all&per_page=100&page={page}")
            prs.extend(batch)
            if len(batch) < 100:
                break
        else:
            raise Conflict("PR pagination exceeded limit; no partial snapshot")
        return {"repository": self.c["repository"], "rows": rows, "issues": issues,
                "prs": [{k: p.get(k) for k in ["number", "title", "body", "state", "merged_at", "html_url"]} for p in prs]}

    def write(self, op):
        if op["kind"] == "jira_due":
            self.request("jira", "/rest/api/3/issue/" + op["key"], "PUT", {"fields": {"duedate": op["after"]}})
        elif op["kind"] == "wbs_note":
            self.request("sheets", ":batchUpdate", "POST", {"requests": [{"updateCells": {
                "range": {"sheetId": self.c["sheet_id"], "startRowIndex": op["row"] - 1,
                          "endRowIndex": op["row"], "startColumnIndex": 7, "endColumnIndex": 8},
                "rows": [{"values": [{"note": op["after"]}]}], "fields": "note"}}]})
        else:
            raise Conflict("Unknown operation")


def apply_plan(c, plan, api, journal):
    # Recompute from live data. Serialized operations are never trusted on their own.
    current = api.snapshot()
    expected = build_plan(c, current)
    if plan != expected:
        raise Conflict("Stale/modified plan; generate a new plan")
    results = []
    for op in plan["operations"]:
        # Recheck global ownership and the exact pending operation immediately before each write.
        fresh = build_plan(c, api.snapshot())
        if op not in fresh["operations"]:
            raise Conflict("Concurrent edit detected; partial results remain in journal")
        api.write(op)
        results.append({"operation": op, "status": "write_returned"})
        save(journal, results)
        after = build_plan(c, api.snapshot())
        identity = "key" if op["kind"] == "jira_due" else "wbs_id"
        if any(o["kind"] == op["kind"] and o[identity] == op[identity] for o in after["operations"]):
            raise Conflict("Write readback failed; inspect journal before retry")
        results[-1]["status"] = "verified"
        save(journal, results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "apply"])
    p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    p.add_argument("--snapshot", help="Offline normalized MCP snapshot (plan only)")
    p.add_argument("--plan", required=True)
    p.add_argument("--journal", default="sync-journal.json")
    args = p.parse_args()
    c = load(args.config)
    validate_config(c)
    api = Api(c)
    if args.command == "plan":
        snapshot = load(args.snapshot) if args.snapshot else api.snapshot()
        plan = build_plan(c, snapshot)
        save(args.plan, plan)
        print(f"Planned {len(plan['operations'])} operations; {len(plan['warnings'])} warnings")
    else:
        if args.snapshot:
            p.error("apply requires live reads; offline snapshot cannot authorize writes")
        lock = Path(args.journal + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError:
            raise Conflict("Another apply is running; inspect stale lock manually") from None
        try:
            print(f"Verified {len(apply_plan(c, load(args.plan), api, args.journal))} operations")
        finally:
            lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except (Conflict, KeyError, OSError) as error:
        print(f"Sync stopped: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
