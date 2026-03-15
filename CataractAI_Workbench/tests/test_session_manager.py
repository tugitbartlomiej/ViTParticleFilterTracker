"""Tests for SessionManager and its helper functions."""

import pytest

from CataractAI_Workbench.app.tabs.experiments.session_manager import (
    SessionManager,
    _extract_title_from_heading,
    _parse_metadata_line,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

FULL_SUMMARY = """\
# Session Summary: DETR Benchmark Run

## Metadata
- **Date:** 2026-01-15
- **Time:** 14:30:00
- **Type:** Benchmark
- **Status:** Completed
- **Topics:** DETR, Query 81, Benchmark

## Objective
Run full DETR benchmark on Q81 dataset with all epochs.

## Actions Taken
1. Configured benchmark parameters
2. Executed benchmark script

## Key Findings
- mAP improved by 3.2% over baseline
- Training converged at epoch 50

## Files Modified
- benchmark_results/results.json
- benchmark_results/plots/

## Next Steps
- [ ] Analyze per-class metrics
- [x] Generate summary report

## Related Sessions
- Session_2026-01-14_120000_DETR_Training
"""

MINIMAL_SUMMARY = """\
# Session Summary: Quick Test

## Metadata
- **Date:** 2026-03-01
- **Type:** Analysis
"""

EMPTY_SUMMARY = ""


@pytest.fixture
def mgr(tmp_path):
    """Return a SessionManager whose SESSIONS_DIR points to *tmp_path*."""
    m = SessionManager()
    m.SESSIONS_DIR = tmp_path
    return m


def _make_session(tmp_path, dir_name, content=None):
    """Helper: create a session directory with an optional SESSION_SUMMARY.md."""
    d = tmp_path / dir_name
    d.mkdir(parents=True, exist_ok=True)
    if content is not None:
        (d / "SESSION_SUMMARY.md").write_text(content, encoding="utf-8")
    return d


# ================================================================== #
# _extract_title_from_heading
# ================================================================== #

class TestExtractTitleFromHeading:
    def test_with_session_summary_prefix(self):
        assert _extract_title_from_heading("Session Summary: DETR Run") == "DETR Run"

    def test_with_session_summary_prefix_different_case(self):
        assert _extract_title_from_heading("session summary: DETR Run") == "DETR Run"

    def test_without_prefix(self):
        assert _extract_title_from_heading("DETR Run") == "DETR Run"

    def test_session_summary_without_colon(self):
        # "Session Summary" (no colon) followed by dash separator
        assert _extract_title_from_heading("Session Summary - My Title") == "My Title"

    def test_session_summary_without_colon_no_dash(self):
        result = _extract_title_from_heading("Session Summary")
        assert result == ""

    def test_whitespace_stripping(self):
        assert _extract_title_from_heading("  Session Summary:  Spaced  ") == "Spaced"

    def test_empty_string(self):
        assert _extract_title_from_heading("") == ""


# ================================================================== #
# _parse_metadata_line
# ================================================================== #

class TestParseMetadataLine:
    def test_date_field(self):
        target = {}
        _parse_metadata_line("- **Date:** 2026-01-15", {"- **Date:**": "date"}, target)
        assert target == {"date": "2026-01-15"}

    def test_type_field(self):
        target = {}
        _parse_metadata_line("- **Type:** Benchmark", {"- **Type:**": "type"}, target)
        assert target == {"type": "Benchmark"}

    def test_status_field(self):
        target = {}
        _parse_metadata_line("- **Status:** Completed", {"- **Status:**": "status"}, target)
        assert target == {"status": "Completed"}

    def test_time_field(self):
        target = {}
        _parse_metadata_line("- **Time:** 14:30:00", {"- **Time:**": "time"}, target)
        assert target == {"time": "14:30:00"}

    def test_topics_field(self):
        """Topics are handled specially regardless of field_map."""
        target = {}
        _parse_metadata_line("- **Topics:** DETR, YOLO, Benchmark", {}, target)
        assert target == {"topics": ["DETR", "YOLO", "Benchmark"]}

    def test_topics_empty_string(self):
        target = {}
        _parse_metadata_line("- **Topics:** ", {}, target)
        # Empty string after stripping -> no topics key added
        assert "topics" not in target or target.get("topics") == []

    def test_no_match(self):
        target = {}
        _parse_metadata_line("Some random line", {"- **Date:**": "date"}, target)
        assert target == {}

    def test_whitespace_preserved(self):
        target = {}
        _parse_metadata_line("  - **Type:** Training  ", {"- **Type:**": "type"}, target)
        assert target == {"type": "Training"}


# ================================================================== #
# parse_session_summary
# ================================================================== #

class TestParseSessionSummary:
    def test_full_content(self, mgr):
        result = mgr.parse_session_summary(FULL_SUMMARY)
        assert result["title"] == "DETR Benchmark Run"
        assert result["description"] == "DETR Benchmark Run"
        assert result["date"] == "2026-01-15"
        assert result["time"] == "14:30:00"
        assert result["type"] == "Benchmark"
        assert result["status"] == "Completed"
        assert result["topics"] == ["DETR", "Query 81", "Benchmark"]
        assert "Q81 dataset" in result["objective"]
        assert "benchmark parameters" in result["actions"]
        assert "mAP improved" in result["findings"]
        assert "benchmark_results" in result["files_modified"]
        assert "per-class metrics" in result["next_steps"]

    def test_minimal_content(self, mgr):
        result = mgr.parse_session_summary(MINIMAL_SUMMARY)
        assert result["title"] == "Quick Test"
        assert result["date"] == "2026-03-01"
        assert result["type"] == "Analysis"
        # Fields not present should be empty defaults
        assert result["time"] == ""
        assert result["status"] == ""
        assert result["objective"] == ""
        assert result["actions"] == ""

    def test_empty_content(self, mgr):
        result = mgr.parse_session_summary(EMPTY_SUMMARY)
        assert result["title"] == ""
        assert result["date"] == ""
        assert result["type"] == ""
        assert result["topics"] == []

    def test_missing_sections(self, mgr):
        content = "# Session Summary: No Sections\n\n## Metadata\n- **Type:** Training\n"
        result = mgr.parse_session_summary(content)
        assert result["title"] == "No Sections"
        assert result["type"] == "Training"
        assert result["objective"] == ""
        assert result["actions"] == ""
        assert result["findings"] == ""
        assert result["files_modified"] == ""
        assert result["next_steps"] == ""


# ================================================================== #
# _extract_section
# ================================================================== #

class TestExtractSection:
    def test_simple_section(self):
        lines = [
            "## Objective",
            "Do the thing.",
            "",
            "## Actions Taken",
            "1. Did it.",
        ]
        result = SessionManager._extract_section(lines, "Objective")
        assert result == "Do the thing."

    def test_multiline_section(self):
        lines = [
            "## Key Findings",
            "- Finding A",
            "- Finding B",
            "",
            "## Next Steps",
        ]
        result = SessionManager._extract_section(lines, "Key Findings")
        assert "Finding A" in result
        assert "Finding B" in result

    def test_section_not_found(self):
        lines = ["## Objective", "Something"]
        assert SessionManager._extract_section(lines, "Nonexistent") == ""

    def test_section_stopped_by_h1(self):
        lines = [
            "## Objective",
            "Content here",
            "# New Top Heading",
            "Should not be included",
        ]
        result = SessionManager._extract_section(lines, "Objective")
        assert result == "Content here"
        assert "Should not" not in result

    def test_case_insensitive_heading_match(self):
        lines = [
            "## KEY FINDINGS",
            "- Result X",
        ]
        result = SessionManager._extract_section(lines, "Key Findings")
        assert "Result X" in result

    def test_empty_section(self):
        lines = [
            "## Objective",
            "",
            "## Actions Taken",
        ]
        result = SessionManager._extract_section(lines, "Objective")
        assert result == ""


# ================================================================== #
# _matches_filters
# ================================================================== #

class TestMatchesFilters:
    def test_no_filters(self):
        info = {"description": "test", "type": "Training"}
        assert SessionManager._matches_filters(info, None, None) is True

    def test_type_filter_all(self):
        info = {"description": "test", "type": "Training"}
        assert SessionManager._matches_filters(info, "All", None) is True

    def test_type_filter_match(self):
        info = {"description": "test", "type": "Benchmark"}
        assert SessionManager._matches_filters(info, "Benchmark", None) is True

    def test_type_filter_no_match(self):
        info = {"description": "test", "type": "Training"}
        assert SessionManager._matches_filters(info, "Benchmark", None) is False

    def test_search_query_in_description(self):
        info = {"description": "DETR test", "type": "", "objective": "",
                "topics": [], "dir_name": ""}
        assert SessionManager._matches_filters(info, None, "detr") is True

    def test_search_query_in_topics(self):
        info = {"description": "", "type": "", "objective": "",
                "topics": ["YOLO", "DETR"], "dir_name": ""}
        assert SessionManager._matches_filters(info, None, "yolo") is True

    def test_search_query_no_match(self):
        info = {"description": "something", "type": "Training", "objective": "",
                "topics": [], "dir_name": "Session_2026"}
        assert SessionManager._matches_filters(info, None, "nonexistent") is False

    def test_both_filters_match(self):
        info = {"description": "DETR run", "type": "Benchmark", "objective": "",
                "topics": [], "dir_name": ""}
        assert SessionManager._matches_filters(info, "Benchmark", "detr") is True

    def test_type_matches_search_fails(self):
        info = {"description": "something", "type": "Benchmark", "objective": "",
                "topics": [], "dir_name": ""}
        assert SessionManager._matches_filters(info, "Benchmark", "nonexistent") is False

    def test_type_filter_case_insensitive(self):
        info = {"description": "", "type": "Fix / Debugging"}
        assert SessionManager._matches_filters(info, "fix", None) is True


# ================================================================== #
# create_session
# ================================================================== #

class TestCreateSession:
    def test_directory_created(self, mgr, tmp_path):
        dir_name = mgr.create_session("My Test", "Training", ["DETR"], "Train model")
        assert (tmp_path / dir_name).is_dir()

    def test_summary_file_created(self, mgr, tmp_path):
        dir_name = mgr.create_session("My Test", "Training", ["DETR"], "Train model")
        summary_path = tmp_path / dir_name / "SESSION_SUMMARY.md"
        assert summary_path.is_file()

    def test_readme_file_created(self, mgr, tmp_path):
        dir_name = mgr.create_session("My Test", "Training", ["DETR"], "Train model")
        readme_path = tmp_path / dir_name / "README.md"
        assert readme_path.is_file()

    def test_summary_content(self, mgr, tmp_path):
        dir_name = mgr.create_session("My Test", "Training", ["DETR", "YOLO"], "Train model")
        content = (tmp_path / dir_name / "SESSION_SUMMARY.md").read_text(encoding="utf-8")
        assert "# Session Summary: My Test" in content
        assert "- **Type:** Training" in content
        assert "- **Status:** In Progress" in content
        assert "DETR, YOLO" in content
        assert "Train model" in content

    def test_readme_content(self, mgr, tmp_path):
        dir_name = mgr.create_session("My Test", "Benchmark", ["DETR"], "Run bench")
        content = (tmp_path / dir_name / "README.md").read_text(encoding="utf-8")
        assert "# My Test" in content
        assert "**Type:** Benchmark" in content
        assert "DETR" in content

    def test_dir_name_format(self, mgr):
        dir_name = mgr.create_session("Hello World", "Training", [], "")
        assert dir_name.startswith("Session_")
        assert "Hello_World" in dir_name

    def test_special_chars_sanitized(self, mgr):
        dir_name = mgr.create_session("Test!@#$%", "Training", [], "")
        # Special chars should be removed
        assert "!" not in dir_name
        assert "@" not in dir_name

    def test_empty_description(self, mgr, tmp_path):
        dir_name = mgr.create_session("", "Training", [], "")
        assert (tmp_path / dir_name).is_dir()
        content = (tmp_path / dir_name / "SESSION_SUMMARY.md").read_text(encoding="utf-8")
        assert "# Session Summary: Untitled" in content

    def test_empty_topics(self, mgr, tmp_path):
        dir_name = mgr.create_session("Test", "Training", [], "Objective")
        content = (tmp_path / dir_name / "SESSION_SUMMARY.md").read_text(encoding="utf-8")
        assert "- **Topics:** \n" in content or "- **Topics:**" in content


# ================================================================== #
# save_session
# ================================================================== #

class TestSaveSession:
    def test_write_and_read(self, mgr, tmp_path):
        dir_name = "Session_2026-03-14_120000_Test"
        (tmp_path / dir_name).mkdir()
        mgr.save_session(dir_name, "# Updated Content\nNew data here.")
        saved = (tmp_path / dir_name / "SESSION_SUMMARY.md").read_text(encoding="utf-8")
        assert saved == "# Updated Content\nNew data here."

    def test_overwrite_existing(self, mgr, tmp_path):
        dir_name = "Session_2026-03-14_120000_Test"
        _make_session(tmp_path, dir_name, "Old content")
        mgr.save_session(dir_name, "New content")
        saved = (tmp_path / dir_name / "SESSION_SUMMARY.md").read_text(encoding="utf-8")
        assert saved == "New content"


# ================================================================== #
# list_sessions
# ================================================================== #

class TestListSessions:
    def test_empty_directory(self, mgr):
        result = mgr.list_sessions()
        assert result == []

    def test_nonexistent_directory(self, tmp_path):
        m = SessionManager()
        m.SESSIONS_DIR = tmp_path / "does_not_exist"
        assert m.list_sessions() == []

    def test_lists_valid_sessions(self, mgr, tmp_path):
        _make_session(tmp_path, "Session_2026-01-10_100000_Alpha",
                      "# Alpha\n## Metadata\n- **Type:** Training\n")
        _make_session(tmp_path, "Session_2026-01-11_120000_Beta",
                      "# Beta\n## Metadata\n- **Type:** Benchmark\n")
        result = mgr.list_sessions()
        assert len(result) == 2

    def test_sorted_newest_first(self, mgr, tmp_path):
        _make_session(tmp_path, "Session_2026-01-10_100000_Old",
                      "# Old\n## Metadata\n- **Type:** Training\n")
        _make_session(tmp_path, "Session_2026-02-15_120000_New",
                      "# New\n## Metadata\n- **Type:** Training\n")
        result = mgr.list_sessions()
        assert result[0]["dir_name"] == "Session_2026-02-15_120000_New"
        assert result[1]["dir_name"] == "Session_2026-01-10_100000_Old"

    def test_ignores_non_session_dirs(self, mgr, tmp_path):
        (tmp_path / "analysis").mkdir()
        (tmp_path / "tools").mkdir()
        _make_session(tmp_path, "Session_2026-01-10_100000_Real",
                      "# Real\n## Metadata\n- **Type:** Training\n")
        result = mgr.list_sessions()
        assert len(result) == 1

    def test_type_filter(self, mgr, tmp_path):
        _make_session(tmp_path, "Session_2026-01-10_100000_A",
                      "# A\n## Metadata\n- **Type:** Training\n")
        _make_session(tmp_path, "Session_2026-01-11_120000_B",
                      "# B\n## Metadata\n- **Type:** Benchmark\n")
        result = mgr.list_sessions(type_filter="Benchmark")
        assert len(result) == 1
        assert result[0]["type"] == "Benchmark"

    def test_search_query(self, mgr, tmp_path):
        _make_session(tmp_path, "Session_2026-01-10_100000_DETR_Work",
                      "# DETR Work\n## Metadata\n- **Type:** Training\n")
        _make_session(tmp_path, "Session_2026-01-11_120000_YOLO_Fix",
                      "# YOLO Fix\n## Metadata\n- **Type:** Fix / Debugging\n")
        result = mgr.list_sessions(search_query="detr")
        assert len(result) == 1
        assert "DETR" in result[0]["description"]

    def test_no_summary_file(self, mgr, tmp_path):
        """Session dir without SESSION_SUMMARY.md should still be listed."""
        _make_session(tmp_path, "Session_2026-01-10_100000_NoSummary")
        result = mgr.list_sessions()
        assert len(result) == 1
        assert result[0]["dir_name"] == "Session_2026-01-10_100000_NoSummary"


# ================================================================== #
# get_session
# ================================================================== #

class TestGetSession:
    def test_existing_session(self, mgr, tmp_path):
        dir_name = "Session_2026-01-15_143000_DETR_Run"
        _make_session(tmp_path, dir_name, FULL_SUMMARY)
        result = mgr.get_session(dir_name)
        assert result["dir_name"] == dir_name
        assert result["title"] == "DETR Benchmark Run"
        assert result["type"] == "Benchmark"
        assert "raw_content" in result
        assert result["path"] == str(tmp_path / dir_name)

    def test_nonexistent_session(self, mgr, tmp_path):
        result = mgr.get_session("Session_2026-99-99_000000_Ghost")
        assert "error" in result

    def test_date_from_dirname_when_missing(self, mgr, tmp_path):
        """When summary has no date, get_session fills it from dir name."""
        dir_name = "Session_2026-05-20_183000_NoDate"
        _make_session(tmp_path, dir_name, "# Just a title\n")
        result = mgr.get_session(dir_name)
        assert result["date"] == "2026-05-20"
        assert result["time"] == "18:30:00"

    def test_description_from_dirname_when_missing(self, mgr, tmp_path):
        dir_name = "Session_2026-05-20_183000_My_Desc"
        _make_session(tmp_path, dir_name, "## Metadata\n- **Type:** Training\n")
        result = mgr.get_session(dir_name)
        assert result["description"] == "My Desc"


# ================================================================== #
# get_statistics
# ================================================================== #

class TestGetStatistics:
    def test_empty(self, mgr):
        stats = mgr.get_statistics()
        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["by_topic"] == {}

    def test_counts_by_type(self, mgr, tmp_path):
        _make_session(tmp_path, "Session_2026-01-10_100000_A",
                      "# A\n## Metadata\n- **Type:** Training\n")
        _make_session(tmp_path, "Session_2026-01-11_110000_B",
                      "# B\n## Metadata\n- **Type:** Training\n")
        _make_session(tmp_path, "Session_2026-01-12_120000_C",
                      "# C\n## Metadata\n- **Type:** Benchmark\n")
        stats = mgr.get_statistics()
        assert stats["total"] == 3
        assert stats["by_type"]["Training"] == 2
        assert stats["by_type"]["Benchmark"] == 1

    def test_counts_by_topic(self, mgr, tmp_path):
        _make_session(
            tmp_path, "Session_2026-01-10_100000_A",
            "# A\n## Metadata\n- **Type:** Training\n- **Topics:** DETR, YOLO\n",
        )
        _make_session(
            tmp_path, "Session_2026-01-11_110000_B",
            "# B\n## Metadata\n- **Type:** Benchmark\n- **Topics:** DETR\n",
        )
        stats = mgr.get_statistics()
        assert stats["by_topic"]["DETR"] == 2
        assert stats["by_topic"]["YOLO"] == 1
