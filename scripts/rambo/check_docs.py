#!/usr/bin/env python3
"""Validate documentation metadata, local links, source maps, and Work-ID lifecycle."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parents[1]))
from check_governance import check as check_governance

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CHECKED_TREES = ("current", "research", "decisions", "changes", "work", "governance", "upstream")
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
ABSOLUTE_LEAK = re.compile(r"(?:file://|/root/|/home/)")


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError("unterminated YAML frontmatter")
    value = yaml.safe_load(text[4:closing])
    if not isinstance(value, dict):
        raise ValueError("frontmatter must be a mapping")
    for field in ("id", "type", "status", "source_map"):
        if field not in value:
            raise ValueError(f"frontmatter missing '{field}'")
    if not isinstance(value["source_map"], list):
        raise ValueError("source_map must be a list")
    return value


def local_link_exists(path: Path, target: str) -> bool:
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return True
    target = target.split("#", 1)[0]
    return bool(target) and (path.parent / target).resolve().exists()


def main() -> int:
    failures: list[str] = []
    documents: list[tuple[Path, dict[str, Any]]] = []
    for tree in CHECKED_TREES:
        for path in (DOCS / tree).rglob("*.md"):
            try:
                metadata = frontmatter(path)
                documents.append((path, metadata))
            except ValueError as error:
                failures.append(f"{path.relative_to(ROOT)}: {error}")
                continue
            text = path.read_text(encoding="utf-8")
            if ABSOLUTE_LEAK.search(text):
                failures.append(f"{path.relative_to(ROOT)}: contains an absolute local path")
            for source in metadata["source_map"]:
                source_path = Path(source)
                if source_path.is_absolute() or ".." in source_path.parts or not (ROOT / source_path).exists():
                    failures.append(f"{path.relative_to(ROOT)}: invalid source_map entry {source!r}")
            link_text = re.sub(r"```[\s\S]*?```", "", text)
            for target in LINK.findall(link_text):
                if not local_link_exists(path, target):
                    failures.append(f"{path.relative_to(ROOT)}: broken local link {target!r}")
            relative_parts = path.relative_to(DOCS).parts
            if "active" in relative_parts and metadata["status"] != "active":
                failures.append(f"{path.relative_to(ROOT)}: active document must use status: active")
            if "archive" in relative_parts and metadata["status"] == "active":
                failures.append(f"{path.relative_to(ROOT)}: archived document cannot use status: active")

    change_ids = {meta["id"] for path, meta in documents if "changes" in path.parts and meta["type"] == "change"}
    plan_ids = {meta["id"] for path, meta in documents if "work" in path.parts and meta["type"] == "exec-plan"}
    if change_ids != plan_ids:
        failures.append(f"ChangeSpec/ExecPlan Work IDs differ: {sorted(change_ids)} vs {sorted(plan_ids)}")
    failures.extend(check_governance(ROOT))
    if failures:
        print("Documentation check failed:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print(f"Documentation: OK ({len(documents)} governed documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
