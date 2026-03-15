"""Widget for generating benchmark reports in JSON, HTML, or CSV format."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...widgets.file_picker import FilePicker


class ReportGenerator(QWidget):
    """Generate benchmark reports in various formats with preview."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._results: List[dict] = []
        self._init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -- Format + output path -------------------------------------------
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Format:"))
        self._combo_format = QComboBox()
        self._combo_format.addItems(["JSON", "HTML", "CSV"])
        self._combo_format.setFixedWidth(80)
        self._combo_format.currentIndexChanged.connect(self._update_preview)
        row1.addWidget(self._combo_format)

        row1.addWidget(QLabel("Output:"))
        self._picker = FilePicker(
            label="Save As",
            mode="save",
            filter_str="All Files (*);;JSON (*.json);;HTML (*.html);;CSV (*.csv)",
        )
        row1.addWidget(self._picker, stretch=1)

        self._btn_generate = QPushButton("Generate Report")
        self._btn_generate.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; font-weight: bold; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._btn_generate.clicked.connect(self._on_generate)
        row1.addWidget(self._btn_generate)

        layout.addLayout(row1)

        # -- Preview area ----------------------------------------------------
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet(
            "QTextEdit { background-color: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #3C3C3C; font-family: Consolas; font-size: 9pt; }"
        )
        layout.addWidget(self._preview, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_results(self, results: List[dict]):
        """Store the results to be exported and refresh preview."""
        self._results = list(results)
        self._update_preview()

    def generate_json(self, results: List[dict], path: str):
        """Save results as a pretty-printed JSON file."""
        report = self._build_report_dict(results)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    def generate_html(self, results: List[dict], path: str):
        """Create a self-contained HTML benchmark report."""
        html = self._build_html(results)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)

    def generate_csv(self, results: List[dict], path: str):
        """Save results as a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if not results:
            return

        # Determine all unique keys
        all_keys = []
        for r in results:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report_dict(results: List[dict]) -> dict:
        """Build a report envelope around the raw results."""
        return {
            "report": {
                "title": "CataractAI Benchmark Report",
                "generated": datetime.now().isoformat(),
                "num_entries": len(results),
            },
            "results": results,
        }

    @staticmethod
    def _build_html(results: List[dict]) -> str:
        """Build a self-contained dark-theme HTML report."""
        # Determine columns
        columns = ["model", "epoch", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall", "f1_score"]
        present_cols = []
        for c in columns:
            if any(c in r for r in results):
                present_cols.append(c)

        # Table rows
        rows_html = ""
        for r in results:
            cells = ""
            for c in present_cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    val = f"{val:.2f}"
                cells += f"<td>{val}</td>"
            rows_html += f"<tr>{cells}</tr>\n"

        header_cells = "".join(f"<th>{c}</th>" for c in present_cols)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CataractAI Benchmark Report</title>
<style>
  body {{
    background: #1E1E1E; color: #D4D4D4; font-family: 'Segoe UI', Arial, sans-serif;
    margin: 20px; padding: 0;
  }}
  h1 {{ color: #4FC3F7; border-bottom: 2px solid #333; padding-bottom: 10px; }}
  h2 {{ color: #81C784; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
  table {{
    border-collapse: collapse; width: 100%; margin: 20px 0;
    background: #252525;
  }}
  th {{
    background: #333; color: #4FC3F7; padding: 10px 12px;
    text-align: center; border: 1px solid #444;
  }}
  td {{
    padding: 8px 12px; text-align: center; border: 1px solid #333;
  }}
  tr:nth-child(even) {{ background: #2A2A2A; }}
  tr:hover {{ background: #333; }}
  .footer {{ margin-top: 30px; color: #666; font-size: 0.8em; text-align: center; }}
</style>
</head>
<body>
<h1>CataractAI Benchmark Report</h1>
<div class="meta">Generated: {timestamp} | Entries: {len(results)}</div>

<h2>Results Summary</h2>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>

<div class="footer">Generated by CataractAI Workbench</div>
</body>
</html>"""
        return html

    def _update_preview(self, _index: int = 0):
        """Refresh the preview area based on the selected format."""
        if not self._results:
            self._preview.setPlainText("(No results loaded)")
            return

        fmt = self._combo_format.currentText()
        if fmt == "JSON":
            report = self._build_report_dict(self._results)
            text = json.dumps(report, indent=2, default=str)
            self._preview.setPlainText(text[:10000])
        elif fmt == "HTML":
            html = self._build_html(self._results)
            self._preview.setPlainText(html[:10000])
        elif fmt == "CSV":
            lines = []
            if self._results:
                all_keys = []
                for r in self._results:
                    for k in r.keys():
                        if k not in all_keys:
                            all_keys.append(k)
                lines.append(",".join(all_keys))
                for r in self._results:
                    lines.append(",".join(str(r.get(k, "")) for k in all_keys))
            self._preview.setPlainText("\n".join(lines)[:10000])

    def _on_generate(self):
        """Generate the report in the selected format."""
        path = self._picker.path()
        if not path:
            self._preview.setPlainText("Error: Please specify an output path.")
            return

        fmt = self._combo_format.currentText()
        try:
            if fmt == "JSON":
                if not path.endswith(".json"):
                    path += ".json"
                self.generate_json(self._results, path)
            elif fmt == "HTML":
                if not path.endswith(".html"):
                    path += ".html"
                self.generate_html(self._results, path)
            elif fmt == "CSV":
                if not path.endswith(".csv"):
                    path += ".csv"
                self.generate_csv(self._results, path)

            self._preview.setPlainText(
                f"Report generated successfully.\n\nFormat: {fmt}\nPath: {path}"
            )
        except Exception as exc:
            self._preview.setPlainText(f"Error generating report:\n{exc}")
