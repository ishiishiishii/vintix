#!/usr/bin/env python3
"""Sync GitHub issues into the Notion「研究イシュー」database (property-typed).

Uses Notion DB/data source (Status / multi-select Labels / select Priority / URL),
not a plain markdown table. Priority stars are preserved if the row already exists;
new OPEN issues default to ★★☆☆☆ unless overridden in DEFAULT_PRIORITY.

Also ensures the parent page has two linked tables filtered by OPEN / CLOSED
(GitHub state → Notion「状態」).

    python autopilot/sync_notion_issues_db.py
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Parent page: 研究イシュー（GitHub 同期）
PARENT_PAGE_ID = "40ed65e6-123f-81a9-80c3-ed7590337e8d"
DATABASE_ID = "41cd65e6-123f-8158-ae6e-ee48ae665864"
DATA_SOURCE_ID = "41cd65e6-123f-81f7-a5af-000b48fff419"
# Linked filtered tables (created once; IDs stored after first run)
STATE_FILE = Path(__file__).resolve().parent / ".notion_issues_views.json"

KNOWN_LABELS = {
    "task", "discussion", "experiment", "needs-decision", "autopilot", "later",
}
DEFAULT_PRIORITY = {
    18: "★★★★★",
    21: "★★★★★",
    11: "★★★★☆",
    22: "★★★★☆",
    20: "★★★☆☆",
    16: "★★☆☆☆",
    19: "★☆☆☆☆",
}
DEFAULT_SUMMARY = {
    21: "多様な体の成功履歴で学習→Go1で重さ・大きさを変えて評価",
    20: "他機種学習のシード差＝学習データ欠落かの切り分け",
    19: "直近外・IROS後段のGo2実機",
    18: "1機種データだけで学習したモデルの体スケール評価",
    22: "汎用歩行の代表手法を調査・導入／比較用",
    16: "評価ロボット拡充（URDF/Env・立位確認）",
    11: "多様な体×学習シードの履歴をまとめて収集",
}


def notion_token() -> str:
    if os.environ.get("NOTION_TOKEN"):
        return os.environ["NOTION_TOKEN"]
    mcp = Path.home() / ".cursor" / "mcp.json"
    data = json.loads(mcp.read_text())
    return data["mcpServers"]["notion"]["env"]["NOTION_TOKEN"]


def api(method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://api.notion.com{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {notion_token()}",
            "Notion-Version": "2025-09-03",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Notion {method} {path}: {e.read().decode()[:800]}") from e


def list_github_issues() -> list[dict]:
    raw = subprocess.check_output(
        [
            "gh", "issue", "list", "-R", "ishiishiishii/vintix",
            "--state", "all", "--limit", "200",
            "--json", "number,title,state,labels,updatedAt,url",
        ],
        text=True,
    )
    return json.loads(raw)


def query_existing() -> dict[int, dict]:
    """Map issue number -> {page_id, priority_name, summary}."""
    out: dict[int, dict] = {}
    cursor = None
    while True:
        payload: dict = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        data = api("POST", f"/v1/data_sources/{DATA_SOURCE_ID}/query", payload)
        for row in data.get("results", []):
            props = row.get("properties", {})
            num = (props.get("#") or {}).get("number")
            if num is None:
                continue
            pri = ((props.get("優先度") or {}).get("select") or {}) or {}
            summ_arr = (props.get("要約") or {}).get("rich_text") or []
            summary = "".join(t.get("plain_text", "") for t in summ_arr)
            out[int(num)] = {
                "page_id": row["id"],
                "priority": pri.get("name"),
                "summary": summary,
            }
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return out


def props_for(it: dict, *, priority: str, summary: str) -> dict:
    labs = [l["name"] for l in (it.get("labels") or []) if l["name"] in KNOWN_LABELS]
    return {
        "Name": {"title": [{"text": {"content": it["title"][:2000]}}]},
        "#": {"number": it["number"]},
        "状態": {"status": {"name": it["state"]}},
        "ラベル": {"multi_select": [{"name": x} for x in labs]},
        "優先度": {"select": {"name": priority}},
        "GitHub": {"url": it["url"]},
        "更新日": {"date": {"start": it["updatedAt"][:10]}},
        "要約": {"rich_text": [{"text": {"content": summary[:2000]}}]},
    }


def _append_heading(text: str) -> str:
    """Append a heading_2 block; return its block id."""
    data = api(
        "PATCH",
        f"/v1/blocks/{PARENT_PAGE_ID}/children",
        {
            "children": [
                {
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": text}}],
                    },
                }
            ]
        },
    )
    return data["results"][0]["id"]


def ensure_filtered_tables() -> dict:
    """Create linked OPEN / CLOSED tables once; reuse IDs from STATE_FILE."""
    state: dict = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            state = {}

    created_any = False
    for key, status, title in (
        ("open", "OPEN", "オープン（進行中）"),
        ("closed", "CLOSED", "クローズ（完了）"),
    ):
        if state.get(key, {}).get("view_id"):
            continue
        heading_id = _append_heading(title)
        view = api(
            "POST",
            "/v1/views",
            {
                "data_source_id": DATA_SOURCE_ID,
                "name": title,
                "type": "table",
                "filter": {
                    "property": "状態",
                    "status": {"equals": status},
                },
                "sorts": [
                    {"property": "優先度", "direction": "descending"},
                    {"property": "#", "direction": "descending"},
                ],
                "create_database": {
                    "parent": {"type": "page_id", "page_id": PARENT_PAGE_ID},
                    "position": {"type": "after_block", "block_id": heading_id},
                    "show_data_source_title": False,
                },
            },
        )
        state[key] = {
            "view_id": view.get("id"),
            "heading_id": heading_id,
            "status": status,
            "name": title,
        }
        created_any = True
        print(f"created linked table: {title} view={view.get('id')}")

    # Tabs on the source DB as well (handy when opening the DB itself)
    if not state.get("tabs_ok"):
        for name, status in (("オープン", "OPEN"), ("クローズ", "CLOSED")):
            try:
                api(
                    "POST",
                    "/v1/views",
                    {
                        "database_id": DATABASE_ID,
                        "data_source_id": DATA_SOURCE_ID,
                        "name": name,
                        "type": "table",
                        "filter": {
                            "property": "状態",
                            "status": {"equals": status},
                        },
                        "sorts": [
                            {"property": "優先度", "direction": "descending"},
                            {"property": "#", "direction": "descending"},
                        ],
                    },
                )
                print(f"created DB tab view: {name}")
            except SystemExit as e:
                # already exists / permission — non-fatal
                print(f"[warn] tab view {name}: {e}")
        state["tabs_ok"] = True
        created_any = True

    if created_any or not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    return state


def update_parent_blurb(*, n_open: int, n_closed: int) -> None:
    """Refresh intro paragraphs on the parent page (no replace_content)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    children = api("GET", f"/v1/blocks/{PARENT_PAGE_ID}/children?page_size=20")
    # Update first two paragraphs if they look like the intro / sync line
    for block in children.get("results", []):
        if block.get("type") != "paragraph":
            continue
        plain = "".join(
            t.get("plain_text", "") for t in block["paragraph"].get("rich_text", [])
        )
        if "Issue 一覧" in plain or "ishiishiishii/vintix" in plain:
            api(
                "PATCH",
                f"/v1/blocks/{block['id']}",
                {
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": (
                                        "GitHub ishiishiishii/vintix の Issue を、"
                                        "このページ内の表に同期しています。"
                                        "下の「オープン」と「クローズ」は GitHub の状態と同じです"
                                        f"（いま OPEN {n_open} / CLOSED {n_closed}）。"
                                    )
                                },
                            }
                        ]
                    }
                },
            )
        elif "最終同期" in plain or "autopilot/sync_notion_issues_db.py" in plain:
            api(
                "PATCH",
                f"/v1/blocks/{block['id']}",
                {
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "最終同期: "},
                                "annotations": {"bold": True},
                            },
                            {
                                "type": "text",
                                "text": {
                                    "content": (
                                        f"{now}（python autopilot/sync_notion_issues_db.py）"
                                    )
                                },
                            },
                        ]
                    }
                },
            )


def main() -> None:
    existing = query_existing()
    issues = list_github_issues()
    # Track every GitHub issue (open + closed) so both tables stay complete
    by_num = {it["number"]: it for it in issues}
    want_nums = set(by_num) | set(existing)

    created = updated = 0
    for n in sorted(want_nums, reverse=True):
        it = by_num.get(n)
        if it is None:
            continue
        prev = existing.get(n)
        priority = (prev or {}).get("priority") or DEFAULT_PRIORITY.get(n, "★★☆☆☆")
        summary = (prev or {}).get("summary") or DEFAULT_SUMMARY.get(n, it["title"][:100])
        props = props_for(it, priority=priority, summary=summary)
        if prev:
            api("PATCH", f"/v1/pages/{prev['page_id']}", {"properties": props})
            updated += 1
        else:
            api(
                "POST",
                "/v1/pages",
                {
                    "parent": {
                        "type": "data_source_id",
                        "data_source_id": DATA_SOURCE_ID,
                    },
                    "properties": props,
                },
            )
            created += 1

    n_open = sum(1 for it in issues if it["state"] == "OPEN")
    n_closed = sum(1 for it in issues if it["state"] == "CLOSED")
    ensure_filtered_tables()
    update_parent_blurb(n_open=n_open, n_closed=n_closed)
    print(
        f"synced: created={created} updated={updated} "
        f"tracked={len(want_nums)} open={n_open} closed={n_closed}"
    )


if __name__ == "__main__":
    main()
