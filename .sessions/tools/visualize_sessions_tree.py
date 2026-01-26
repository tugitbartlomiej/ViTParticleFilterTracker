"""
2D Session Tree/Timeline Visualization
Generates interactive 2D tree visualization with timeline.
"""

import io
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

SESSIONS_PATH = Path(__file__).parent.parent


def parse_session(session_dir: Path) -> dict | None:
    """Parse session and extract metadata."""
    summary_path = session_dir / "SESSION_SUMMARY.md"
    readme_path = session_dir / "README.md"

    content = ""
    if summary_path.exists():
        content = summary_path.read_text(encoding='utf-8', errors='ignore')
    elif readme_path.exists():
        content = readme_path.read_text(encoding='utf-8', errors='ignore')
    else:
        return None

    folder_name = session_dir.name

    # Parse date
    date_match = re.search(r'Session_(\d{4}-\d{2}-\d{2})_(\d{6})', folder_name)
    if date_match:
        date_str = f"{date_match.group(1)} {date_match.group(2)}"
        try:
            session_date = datetime.strptime(date_str, "%Y-%m-%d %H%M%S")
        except:
            session_date = datetime.now()
    else:
        session_date = datetime.now()

    # Extract name
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder_name)
    name = desc_match.group(1).replace('_', ' ') if desc_match else folder_name[-20:]

    # Parse type
    session_type = "Mixed"
    type_match = re.search(r'\*\*Type:\*\*\s*(.+?)(?:\n|$)', content)
    if type_match:
        type_str = type_match.group(1).strip()
        for t in ["Training", "Benchmark", "Analysis", "SSH", "Writing"]:
            if t in type_str:
                session_type = t
                break

    # Extract objective
    objective = ""
    obj_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if obj_match:
        objective = obj_match.group(1).strip()[:150]

    # Detect main topic/category
    categories = []
    if re.search(r'\bIEEE\b|\bArticle\b|\bPaper\b', content, re.IGNORECASE):
        categories.append("IEEE Article")
    if re.search(r'\bDataset\s*Selection\b|\bEL2N\b|\bDINO\b.*embed', content, re.IGNORECASE):
        categories.append("Dataset Selection")
    if re.search(r'\bYOLO\b', content, re.IGNORECASE):
        categories.append("YOLO")
    if re.search(r'\bDETR\b', content, re.IGNORECASE):
        categories.append("DETR")
    if re.search(r'\bSSH\b|\bEden\b|\bSlurm\b', content, re.IGNORECASE):
        categories.append("SSH/Eden")
    if re.search(r'\bBenchmark\b', content, re.IGNORECASE):
        categories.append("Benchmark")
    if not categories:
        categories.append("Other")

    return {
        "id": folder_name,
        "name": name,
        "date": session_date,
        "type": session_type,
        "objective": objective,
        "categories": categories,
        "main_category": categories[0]
    }


def load_sessions() -> list[dict]:
    """Load all sessions."""
    sessions = []
    for session_dir in SESSIONS_PATH.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith('Session_'):
            session = parse_session(session_dir)
            if session:
                sessions.append(session)
    return sorted(sessions, key=lambda s: s["date"])


def create_tree_visualization(sessions: list[dict], output_path: Path):
    """Create 2D tree/timeline visualization."""

    # Group by main category
    categories = ["IEEE Article", "Dataset Selection", "DETR", "YOLO", "SSH/Eden", "Benchmark", "Other"]
    category_colors = {
        "IEEE Article": "#FFEAA7",
        "Dataset Selection": "#74B9FF",
        "DETR": "#FF7675",
        "YOLO": "#55EFC4",
        "SSH/Eden": "#A29BFE",
        "Benchmark": "#FD79A8",
        "Other": "#B2BEC3"
    }

    # Assign Y positions based on category
    category_y = {cat: i for i, cat in enumerate(categories)}

    # Prepare data
    x_dates = []
    y_positions = []
    colors = []
    texts = []
    hover_texts = []

    for s in sessions:
        x_dates.append(s["date"])
        y_positions.append(category_y.get(s["main_category"], 6))
        colors.append(category_colors.get(s["main_category"], "#B2BEC3"))

        # Short label
        short_name = s["name"][:25] + "..." if len(s["name"]) > 25 else s["name"]
        texts.append(short_name)

        # Hover text
        hover = (
            f"<b>{s['name']}</b><br>"
            f"Date: {s['date'].strftime('%Y-%m-%d %H:%M')}<br>"
            f"Type: {s['type']}<br>"
            f"Categories: {', '.join(s['categories'])}<br>"
            f"<br>{s['objective'][:200]}..."
        )
        hover_texts.append(hover)

    # Create figure
    fig = go.Figure()

    # Add session points
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_positions,
        mode='markers+text',
        marker=dict(
            size=15,
            color=colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=texts,
        textposition='top center',
        textfont=dict(size=9),
        hoverinfo='text',
        hovertext=hover_texts,
        name='Sessions'
    ))

    # Add connecting lines within same category (timeline)
    for cat in categories:
        cat_sessions = [s for s in sessions if s["main_category"] == cat]
        if len(cat_sessions) > 1:
            cat_dates = [s["date"] for s in cat_sessions]
            cat_y = [category_y[cat]] * len(cat_sessions)
            fig.add_trace(go.Scatter(
                x=cat_dates,
                y=cat_y,
                mode='lines',
                line=dict(
                    color=category_colors.get(cat, "#B2BEC3"),
                    width=2,
                    dash='dot'
                ),
                hoverinfo='none',
                showlegend=False
            ))

    # Add category labels on Y axis
    fig.update_layout(
        title=dict(
            text="<b>Session Timeline by Topic</b>",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            type='date'
        ),
        yaxis=dict(
            title="Topic",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            tickmode='array',
            tickvals=list(range(len(categories))),
            ticktext=categories,
            range=[-0.5, len(categories) - 0.5]
        ),
        plot_bgcolor='rgb(250, 250, 255)',
        paper_bgcolor='white',
        height=700,
        margin=dict(l=150, r=50, t=80, b=50),
        hovermode='closest'
    )

    # Add legend for session types
    for cat, color in category_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color),
            name=cat,
            showlegend=True
        ))

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    # Save
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"Saved to: {output_path}")


def create_tree_hierarchical(sessions: list[dict], output_path: Path):
    """Create hierarchical tree view (sunburst/treemap style)."""

    # Build hierarchy: Type -> Category -> Session
    data = []

    # Root
    data.append(dict(
        id="root",
        parent="",
        label="All Sessions",
        value=len(sessions)
    ))

    # Group by type first
    by_type = defaultdict(list)
    for s in sessions:
        by_type[s["type"]].append(s)

    type_colors = {
        "Training": "#FF6B6B",
        "Benchmark": "#4ECDC4",
        "Analysis": "#45B7D1",
        "SSH": "#96CEB4",
        "Writing": "#FFEAA7",
        "Mixed": "#DDA0DD"
    }

    for session_type, type_sessions in by_type.items():
        type_id = f"type_{session_type}"
        data.append(dict(
            id=type_id,
            parent="root",
            label=f"{session_type} ({len(type_sessions)})",
            value=len(type_sessions),
            color=type_colors.get(session_type, "#999")
        ))

        # Group by month within type
        by_month = defaultdict(list)
        for s in type_sessions:
            month_key = s["date"].strftime("%Y-%m")
            by_month[month_key].append(s)

        for month, month_sessions in sorted(by_month.items()):
            month_id = f"{type_id}_{month}"
            data.append(dict(
                id=month_id,
                parent=type_id,
                label=f"{month} ({len(month_sessions)})",
                value=len(month_sessions)
            ))

            # Individual sessions
            for s in month_sessions:
                short_name = s["name"][:30] + "..." if len(s["name"]) > 30 else s["name"]
                data.append(dict(
                    id=s["id"],
                    parent=month_id,
                    label=short_name,
                    value=1,
                    hover=f"{s['name']}<br>{s['date'].strftime('%Y-%m-%d')}<br>{s['objective'][:100]}"
                ))

    # Create sunburst
    ids = [d["id"] for d in data]
    labels = [d["label"] for d in data]
    parents = [d["parent"] for d in data]
    values = [d.get("value", 1) for d in data]
    hovers = [d.get("hover", d["label"]) for d in data]

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        hovertext=hovers,
        hoverinfo="text",
        branchvalues="total",
        insidetextorientation='radial'
    ))

    fig.update_layout(
        title=dict(
            text="<b>Session Hierarchy</b><br><sup>Click to zoom in</sup>",
            x=0.5,
            font=dict(size=20)
        ),
        height=800,
        margin=dict(t=80, l=0, r=0, b=0)
    )

    output_sunburst = output_path.with_name("session_tree_sunburst.html")
    fig.write_html(str(output_sunburst), include_plotlyjs=True, full_html=True)
    print(f"Saved sunburst to: {output_sunburst}")


def main():
    print("Loading sessions...")
    sessions = load_sessions()
    print(f"Loaded {len(sessions)} sessions")

    output_dir = SESSIONS_PATH / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timeline view
    print("\nCreating timeline view...")
    create_tree_visualization(sessions, output_dir / "session_timeline_2d.html")

    # Sunburst/hierarchy view
    print("\nCreating hierarchy view...")
    create_tree_hierarchical(sessions, output_dir / "session_tree_sunburst.html")

    print("\nDone!")


if __name__ == "__main__":
    main()
