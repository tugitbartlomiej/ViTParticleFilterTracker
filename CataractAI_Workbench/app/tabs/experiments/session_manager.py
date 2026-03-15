"""Backend for session management -- list, parse, create, save sessions."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ...core.project_paths import PROJECT_ROOT

# Shared metadata field prefixes used for parsing SESSION_SUMMARY.md
_ALL_FIELDS = {
    "- **Date:**": "date",
    "- **Time:**": "time",
    "- **Type:**": "type",
    "- **Status:**": "status",
}
_QUICK_FIELDS = {"- **Type:**": "type", "- **Status:**": "status"}


def _extract_title_from_heading(heading: str) -> str:
    """Strip 'Session Summary:' prefix from a heading line."""
    title = heading.strip()
    lower = title.lower()
    if lower.startswith("session summary:"):
        title = title[len("Session Summary:"):].strip()
    elif lower.startswith("session summary"):
        title = title[len("Session Summary"):].strip()
        if title.startswith("-"):
            title = title[1:].strip()
    return title


def _parse_metadata_line(line: str, field_map: dict, target: dict) -> None:
    """Match *line* against *field_map* prefixes and store values in *target*."""
    stripped = line.strip()
    for prefix, key in field_map.items():
        if stripped.startswith(prefix):
            target[key] = stripped.split(":**", 1)[-1].strip()
            return
    if stripped.startswith("- **Topics:**"):
        topics_str = stripped.split(":**", 1)[-1].strip()
        if topics_str:
            target["topics"] = [t.strip() for t in topics_str.split(",") if t.strip()]


class SessionManager:
    """Manages experiment sessions stored in .sessions/ directory."""

    SESSIONS_DIR = PROJECT_ROOT / ".sessions"

    SESSION_TYPES = [
        "Training", "Benchmark", "Analysis", "SSH", "Writing", "Mixed",
        "Fix / Debugging", "Development",
    ]

    KNOWN_TOPICS = [
        "DETR", "YOLO", "Dataset Selection", "SSH Eden", "GPU", "Pipeline",
        "IEEE", "Fourier", "EL2N", "DINO", "Query 81", "SAM", "K-Center",
        "K-Means", "Benchmark", "Visualization", "RAG", "CataractAI Workbench",
    ]

    _DIR_PATTERN = re.compile(
        r"^Session_(\d{4}-\d{2}-\d{2})_(\d{6})(?:_(.+))?$"
    )

    def list_sessions(
        self,
        type_filter: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[dict]:
        """Return session dicts sorted by date (newest first)."""
        if not self.SESSIONS_DIR.is_dir():
            return []

        sessions: List[dict] = []
        for entry in self.SESSIONS_DIR.iterdir():
            if not entry.is_dir() or not entry.name.startswith("Session_"):
                continue
            info = self._quick_parse(entry)
            if info is None:
                continue
            if not self._matches_filters(info, type_filter, search_query):
                continue
            sessions.append(info)

        sessions.sort(key=lambda s: s.get("sort_key", ""), reverse=True)
        return sessions

    def get_session(self, session_dir: str) -> dict:
        """Parse SESSION_SUMMARY.md for a given session directory name."""
        dir_path = self.SESSIONS_DIR / session_dir
        summary_path = dir_path / "SESSION_SUMMARY.md"

        if not summary_path.is_file():
            return {"dir_name": session_dir, "error": "SESSION_SUMMARY.md not found"}

        content = summary_path.read_text(encoding="utf-8", errors="replace")
        parsed = self.parse_session_summary(content)
        parsed["dir_name"] = session_dir
        parsed["path"] = str(dir_path)
        parsed["raw_content"] = content

        m = self._DIR_PATTERN.match(session_dir)
        if m:
            if not parsed.get("date"):
                parsed["date"] = m.group(1)
            if not parsed.get("time"):
                t = m.group(2)
                parsed["time"] = f"{t[:2]}:{t[2:4]}:{t[4:6]}"
            if not parsed.get("description") and m.group(3):
                parsed["description"] = m.group(3).replace("_", " ")
        return parsed

    def create_session(
        self,
        description: str,
        session_type: str,
        topics: List[str],
        objective: str,
    ) -> str:
        """Create a new session directory with template files. Returns dir name."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        time_display = now.strftime("%H:%M:%S")

        safe_desc = re.sub(r"[^\w\s-]", "", description).strip()
        safe_desc = re.sub(r"[\s]+", "_", safe_desc)
        dir_name = f"Session_{date_str}_{time_str}"
        if safe_desc:
            dir_name += f"_{safe_desc}"

        dir_path = self.SESSIONS_DIR / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

        topics_str = ", ".join(topics) if topics else ""
        self._write_template(dir_path, description, dir_name,
                             date_str, time_display, session_type,
                             topics_str, objective)
        return dir_name

    def save_session(self, session_dir: str, content: str) -> None:
        """Write content back to SESSION_SUMMARY.md."""
        path = self.SESSIONS_DIR / session_dir / "SESSION_SUMMARY.md"
        path.write_text(content, encoding="utf-8")

    def parse_session_summary(self, content: str) -> dict:
        """Extract structured metadata from SESSION_SUMMARY.md content."""
        result: Dict = {
            "title": "", "date": "", "time": "", "type": "", "status": "",
            "topics": [], "objective": "", "actions": "", "findings": "",
            "files_modified": "", "next_steps": "", "description": "",
        }
        lines = content.split("\n")

        for line in lines:
            if line.startswith("# "):
                title = _extract_title_from_heading(line[2:])
                result["title"] = title
                result["description"] = title
                break

        for line in lines:
            _parse_metadata_line(line, _ALL_FIELDS, result)

        result["objective"] = self._extract_section(lines, "Objective")
        result["actions"] = (self._extract_section(lines, "Actions Taken")
                             or self._extract_section(lines, "Actions"))
        result["findings"] = (self._extract_section(lines, "Key Findings")
                              or self._extract_section(lines, "Results"))
        result["files_modified"] = (
            self._extract_section(lines, "Files Modified")
            or self._extract_section(lines, "Files Generated/Modified"))
        result["next_steps"] = self._extract_section(lines, "Next Steps")
        return result

    def get_statistics(self) -> dict:
        """Return counts by type and topic, plus total session count."""
        sessions = self.list_sessions()
        type_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}

        for s in sessions:
            type_key = s.get("type", "Unknown").split("/")[0].split("|")[0].strip()
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            for topic in s.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        type_counts = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
        topic_counts = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
        return {"total": len(sessions), "by_type": type_counts, "by_topic": topic_counts}

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_section(lines: List[str], heading: str) -> str:
        """Extract text under a given ## heading until the next heading."""
        collecting = False
        section_lines: List[str] = []
        for line in lines:
            if line.startswith("## ") or line.startswith("# "):
                if collecting:
                    break
                if heading.lower() in line.lstrip("#").strip().lower():
                    collecting = True
                    continue
            elif collecting:
                section_lines.append(line)
        return "\n".join(section_lines).strip()

    @staticmethod
    def _matches_filters(info: dict, type_filter: Optional[str],
                         search_query: Optional[str]) -> bool:
        """Return True if a session matches the given filter and search."""
        if type_filter and type_filter != "All":
            if type_filter.lower() not in info.get("type", "").lower():
                return False
        if search_query:
            searchable = " ".join([
                info.get("description", ""), info.get("type", ""),
                info.get("objective", ""), " ".join(info.get("topics", [])),
                info.get("dir_name", ""),
            ]).lower()
            if search_query.lower() not in searchable:
                return False
        return True

    def _quick_parse(self, dir_path: Path) -> Optional[dict]:
        """Quick-parse a session directory for listing purposes."""
        m = self._DIR_PATTERN.match(dir_path.name)
        if not m:
            return None
        date_str = m.group(1)
        time_raw = m.group(2)
        info = {
            "dir_name": dir_path.name,
            "date": date_str,
            "time": f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:6]}",
            "sort_key": f"{date_str}_{time_raw}",
            "description": (m.group(3) or "").replace("_", " "),
            "type": "", "status": "", "topics": [], "objective": "",
            "path": str(dir_path),
        }
        self._enrich_from_summary(dir_path / "SESSION_SUMMARY.md", info)
        return info

    @staticmethod
    def _enrich_from_summary(summary_path: Path, info: dict) -> None:
        """Read first ~40 lines of SESSION_SUMMARY.md to populate metadata."""
        if not summary_path.is_file():
            return
        try:
            with open(summary_path, "r", encoding="utf-8", errors="replace") as fh:
                head = [line.rstrip("\n") for i, line in enumerate(fh) if i < 40]
        except OSError:
            return

        for line in head:
            _parse_metadata_line(line, _QUICK_FIELDS, info)
            stripped = line.strip()
            if stripped.startswith("# ") and not info["description"]:
                title = _extract_title_from_heading(stripped[2:])
                if title:
                    info["description"] = title

        in_objective = False
        for line in head:
            if line.strip().startswith("## Objective"):
                in_objective = True
                continue
            if in_objective:
                if line.strip().startswith("## "):
                    break
                if line.strip():
                    info["objective"] = line.strip()
                    break

    @staticmethod
    def _write_template(dir_path: Path, description: str, dir_name: str,
                        date_str: str, time_display: str, session_type: str,
                        topics_str: str, objective: str) -> None:
        """Write SESSION_SUMMARY.md and README.md templates."""
        summary = f"""# Session Summary: {description or 'Untitled'}

## Metadata
- **Date:** {date_str}
- **Time:** {time_display}
- **Type:** {session_type}
- **Status:** In Progress
- **Topics:** {topics_str}

## Objective
{objective or '[Describe the objective of this session]'}

## Actions Taken
1.

## Key Findings
-

## Files Modified
-

## Next Steps
- [ ]

## Related Sessions
-

---
*Session created: {date_str} {time_display}*
*Project: ViTParticleFilterTracker*
"""
        (dir_path / "SESSION_SUMMARY.md").write_text(summary, encoding="utf-8")

        readme = f"""# {description or dir_name}

**Type:** {session_type}
**Date:** {date_str}
**Topics:** {topics_str}

{objective or ''}

See SESSION_SUMMARY.md for full details.
"""
        (dir_path / "README.md").write_text(readme, encoding="utf-8")
