"""
3D Session Knowledge Graph Visualization
Generates interactive 3D visualization of project sessions using Plotly.

Uses:
- TF-IDF embeddings for X,Y positioning (lightweight, no GPU needed)
- Time as Z axis
- Colors for session types
- Connections for related sessions
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Visualization
import plotly.graph_objects as go

# Text processing & Dimensionality reduction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Session:
    """Represents a single session with its metadata."""
    id: str
    name: str
    date: datetime
    session_type: str = "Mixed"
    status: str = "Unknown"
    objective: str = ""
    summary_text: str = ""
    key_topics: list = field(default_factory=list)
    related_sessions: list = field(default_factory=list)
    files_modified: list = field(default_factory=list)

    # Computed fields
    embedding: Optional[np.ndarray] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def parse_session_summary(session_path: Path) -> Optional[Session]:
    """Parse SESSION_SUMMARY.md and extract metadata."""
    summary_file = session_path / "SESSION_SUMMARY.md"
    readme_file = session_path / "README.md"

    # Try SESSION_SUMMARY.md first, then README.md
    content = ""
    if summary_file.exists():
        content = summary_file.read_text(encoding='utf-8', errors='ignore')
    elif readme_file.exists():
        content = readme_file.read_text(encoding='utf-8', errors='ignore')
    else:
        return None

    # Extract session name from folder
    folder_name = session_path.name

    # Parse date from folder name: Session_YYYY-MM-DD_HHMMSS or Session_YYYY-MM-DD_HHMMSS_Description
    date_match = re.search(r'Session_(\d{4}-\d{2}-\d{2})_(\d{6})', folder_name)
    if date_match:
        date_str = f"{date_match.group(1)} {date_match.group(2)}"
        try:
            session_date = datetime.strptime(date_str, "%Y-%m-%d %H%M%S")
        except:
            session_date = datetime.now()
    else:
        session_date = datetime.now()

    # Extract description from folder name if present
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder_name)
    description = desc_match.group(1).replace('_', ' ') if desc_match else folder_name

    # Parse session type from content
    session_type = "Mixed"
    type_match = re.search(r'\*\*Type:\*\*\s*(.+?)(?:\n|$)', content)
    if type_match:
        type_str = type_match.group(1).strip()
        if 'Training' in type_str:
            session_type = "Training"
        elif 'Benchmark' in type_str:
            session_type = "Benchmark"
        elif 'Analysis' in type_str or 'Research' in type_str:
            session_type = "Analysis"
        elif 'SSH' in type_str:
            session_type = "SSH"
        elif 'Writing' in type_str or 'Article' in type_str:
            session_type = "Writing"

    # Parse status
    status = "Unknown"
    status_match = re.search(r'\*\*Status:\*\*\s*(.+?)(?:\n|$)', content)
    if status_match:
        status = status_match.group(1).strip()

    # Parse objective
    objective = ""
    obj_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if obj_match:
        objective = obj_match.group(1).strip()[:500]  # Limit length

    # Extract key topics from content (look for technical terms)
    topics = set()
    topic_patterns = [
        r'\b(DETR|YOLO|DINO|SAM|EL2N|Fourier|K-Means|K-Center)\b',
        r'\b(IEEE|Article|Paper|Writing)\b',
        r'\b(Dataset|Selection|Pipeline|Training)\b',
        r'\b(Benchmark|Comparison|Analysis)\b',
        r'\b(SSH|Eden|Cluster|GPU)\b',
        r'\b(Query\s*\d+|Q\d+)\b',
    ]
    for pattern in topic_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        topics.update([m.upper() if len(m) <= 5 else m.title() for m in matches])

    # Parse related sessions
    related = []
    related_match = re.search(r'## Related.*?\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if related_match:
        session_refs = re.findall(r'Session_\d{4}-\d{2}-\d{2}_\d{6}(?:_\w+)?', related_match.group(1))
        related = session_refs

    # Also look for session references anywhere in content
    all_refs = re.findall(r'Session_\d{4}-\d{2}-\d{2}_\d{6}(?:_\w+)?', content)
    for ref in all_refs:
        if ref != folder_name and ref not in related:
            related.append(ref)

    # Parse files modified
    files = []
    files_match = re.search(r'## Files.*?\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if files_match:
        file_refs = re.findall(r'[`"]([^`"]+\.(?:py|md|tex|yaml|json))[`"]', files_match.group(1))
        files = file_refs[:10]  # Limit

    return Session(
        id=folder_name,
        name=description,
        date=session_date,
        session_type=session_type,
        status=status,
        objective=objective,
        summary_text=content[:2000],  # Limit for embedding
        key_topics=list(topics)[:15],
        related_sessions=related[:5],
        files_modified=files,
    )


def load_all_sessions(sessions_dir: Path) -> list[Session]:
    """Load all sessions from the .sessions directory."""
    sessions = []

    for item in sessions_dir.iterdir():
        if item.is_dir() and item.name.startswith('Session_'):
            session = parse_session_summary(item)
            if session:
                sessions.append(session)
                print(f"  Loaded: {session.name[:50]}...")

    # Sort by date
    sessions.sort(key=lambda s: s.date)
    return sessions


def compute_embeddings(sessions: list[Session]) -> np.ndarray:
    """Compute TF-IDF embeddings for session content (lightweight, no GPU)."""
    print("\nComputing TF-IDF embeddings...")

    # Create text representation for each session
    texts = []
    for s in sessions:
        text = f"{s.name}. {s.objective}. Topics: {', '.join(s.key_topics)}. {s.summary_text[:1000]}"
        texts.append(text)

    # TF-IDF vectorization with n-grams
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.95,
    )

    print(f"Vectorizing {len(texts)} sessions...")
    embeddings = vectorizer.fit_transform(texts).toarray()

    # Get top terms for debugging
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Top terms: {', '.join(feature_names[:20])}")

    return embeddings


def compute_3d_positions(sessions: list[Session], embeddings: np.ndarray) -> None:
    """Compute 3D positions using t-SNE/PCA for X,Y and time for Z."""
    print("\nComputing 3D positions...")

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)

    # Use PCA if few sessions, t-SNE if many
    if len(sessions) < 10:
        print("Using PCA (few sessions)")
        pca = PCA(n_components=2, random_state=42)
        positions_2d = pca.fit_transform(embeddings_normalized)
    else:
        print("Using t-SNE with optimized params")
        # Higher perplexity = more global structure, better spread
        perplexity = min(30, max(5, len(sessions) // 2))
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity,
            early_exaggeration=12,
            random_state=42
        )
        positions_2d = tsne.fit_transform(embeddings_normalized)

    # Normalize to [-2, 2] range for better spread
    positions_2d = (positions_2d - positions_2d.min(axis=0)) / (positions_2d.max(axis=0) - positions_2d.min(axis=0) + 1e-8)
    positions_2d = positions_2d * 4 - 2  # Scale to [-2, 2]

    # Add small jitter to prevent overlapping points
    np.random.seed(42)
    jitter = np.random.normal(0, 0.08, positions_2d.shape)
    positions_2d += jitter

    # Time as Z axis - scale to [0, 3] for more depth
    dates = [s.date.timestamp() for s in sessions]
    min_date, max_date = min(dates), max(dates)
    date_range = max_date - min_date if max_date != min_date else 1

    for i, session in enumerate(sessions):
        session.embedding = embeddings[i]
        session.x = float(positions_2d[i, 0])
        session.y = float(positions_2d[i, 1])
        session.z = ((session.date.timestamp() - min_date) / date_range) * 3  # Scale to [0, 3]


def create_3d_visualization(sessions: list[Session], output_path: Path) -> None:
    """Create interactive 3D Plotly visualization."""
    print("\nCreating 3D visualization...")

    # Color mapping for session types
    type_colors = {
        "Training": "#FF6B6B",      # Red
        "Benchmark": "#4ECDC4",     # Teal
        "Analysis": "#45B7D1",      # Blue
        "SSH": "#96CEB4",           # Green
        "Writing": "#FFEAA7",       # Yellow
        "Mixed": "#DDA0DD",         # Plum
    }

    # Prepare data for scatter plot
    x_vals = [s.x for s in sessions]
    y_vals = [s.y for s in sessions]
    z_vals = [s.z for s in sessions]
    colors = [type_colors.get(s.session_type, "#DDA0DD") for s in sessions]

    # Create hover text
    hover_texts = []
    for s in sessions:
        topics_str = ", ".join(s.key_topics[:5]) if s.key_topics else "N/A"
        hover_text = (
            f"<b>{s.name}</b><br>"
            f"Date: {s.date.strftime('%Y-%m-%d %H:%M')}<br>"
            f"Type: {s.session_type}<br>"
            f"Status: {s.status}<br>"
            f"Topics: {topics_str}<br>"
            f"<br><i>{s.objective[:150]}...</i>"
        )
        hover_texts.append(hover_text)

    # Create main scatter trace
    scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers+text',
        marker=dict(
            size=14,
            color=colors,
            opacity=0.9,
            line=dict(width=2, color='white'),
            symbol='circle',
        ),
        text=[s.date.strftime('%m/%d') for s in sessions],
        textposition='top center',
        textfont=dict(size=8, color='white'),
        hoverinfo='text',
        hovertext=hover_texts,
        name='Sessions',
    )

    # Create connection lines for related sessions
    edge_x, edge_y, edge_z = [], [], []
    session_map = {s.id: s for s in sessions}

    for session in sessions:
        for related_id in session.related_sessions:
            if related_id in session_map:
                related = session_map[related_id]
                # Add line from session to related
                edge_x.extend([session.x, related.x, None])
                edge_y.extend([session.y, related.y, None])
                edge_z.extend([session.z, related.z, None])

    edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.3)', width=2),
        hoverinfo='none',
        name='Connections',
    )

    # Create figure
    fig = go.Figure(data=[edges, scatter])

    # Add session type legend as annotations
    legend_items = []
    for i, (type_name, color) in enumerate(type_colors.items()):
        legend_items.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=type_name,
                showlegend=True,
            )
        )

    for item in legend_items:
        fig.add_trace(item)

    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Session Knowledge Graph 3D</b><br><sup>X,Y: Semantic Similarity (t-SNE) | Z: Time</sup>",
            x=0.5,
            font=dict(size=20),
        ),
        scene=dict(
            xaxis=dict(
                title="Semantic X",
                showgrid=True,
                gridcolor='rgba(100,100,100,0.3)',
                showbackground=True,
                backgroundcolor='rgb(20, 20, 30)',
            ),
            yaxis=dict(
                title="Semantic Y",
                showgrid=True,
                gridcolor='rgba(100,100,100,0.3)',
                showbackground=True,
                backgroundcolor='rgb(20, 25, 30)',
            ),
            zaxis=dict(
                title="Time →",
                showgrid=True,
                gridcolor='rgba(100,100,100,0.3)',
                showbackground=True,
                backgroundcolor='rgb(20, 20, 35)',
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5),
        ),
        paper_bgcolor='rgb(10, 10, 20)',
        plot_bgcolor='rgb(10, 10, 20)',
        font=dict(color='white'),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(30,30,40,0.8)',
            bordercolor='white',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=900,
    )

    # Save as HTML
    fig.write_html(str(output_path), include_plotlyjs='cdn', full_html=True)
    print(f"\nVisualization saved to: {output_path}")

    # Also create a summary JSON
    summary = {
        "total_sessions": len(sessions),
        "date_range": {
            "start": min(s.date for s in sessions).isoformat(),
            "end": max(s.date for s in sessions).isoformat(),
        },
        "session_types": {t: sum(1 for s in sessions if s.session_type == t) for t in type_colors},
        "all_topics": list(set(topic for s in sessions for topic in s.key_topics)),
        "sessions": [
            {
                "id": s.id,
                "name": s.name,
                "date": s.date.isoformat(),
                "type": s.session_type,
                "topics": s.key_topics,
                "x": s.x,
                "y": s.y,
                "z": s.z,
            }
            for s in sessions
        ]
    }

    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("  SESSION KNOWLEDGE GRAPH 3D VISUALIZATION")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).parent
    sessions_dir = script_dir.parent  # .sessions/
    output_path = sessions_dir / "analysis" / "session_graph_3d.html"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load sessions
    print(f"\nLoading sessions from: {sessions_dir}")
    sessions = load_all_sessions(sessions_dir)
    print(f"\nLoaded {len(sessions)} sessions")

    if len(sessions) < 3:
        print("Error: Need at least 3 sessions for visualization")
        return

    # Compute embeddings
    embeddings = compute_embeddings(sessions)

    # Compute 3D positions
    compute_3d_positions(sessions, embeddings)

    # Create visualization
    create_3d_visualization(sessions, output_path)

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"\nOpen in browser: {output_path}")


if __name__ == "__main__":
    main()
