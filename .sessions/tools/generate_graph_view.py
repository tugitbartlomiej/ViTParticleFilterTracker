"""
Generate interactive graph view like Obsidian - pure HTML/JS, no install needed.
Uses vis.js for force-directed graph.
"""

import io
import re
import json
import sys
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SESSIONS_PATH = Path(__file__).parent.parent


def parse_session(session_dir: Path) -> dict | None:
    """Parse session for graph."""
    summary = session_dir / "SESSION_SUMMARY.md"
    readme = session_dir / "README.md"

    content = ""
    if summary.exists():
        content = summary.read_text(encoding='utf-8', errors='ignore')
    elif readme.exists():
        content = readme.read_text(encoding='utf-8', errors='ignore')
    else:
        return None

    folder = session_dir.name

    # Date
    date_match = re.search(r'Session_(\d{4}-\d{2}-\d{2})_(\d{6})', folder)
    date_str = date_match.group(1) if date_match else "Unknown"

    # Name/Label
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder)
    name = desc_match.group(1).replace('_', ' ') if desc_match else folder
    label = generate_label(name, content)

    # Type
    session_type = "Mixed"
    type_match = re.search(r'\*\*Type:\*\*\s*(.+?)(?:\n|$)', content)
    if type_match:
        t = type_match.group(1)
        for typ in ["Training", "Benchmark", "Analysis", "SSH", "Writing"]:
            if typ in t:
                session_type = typ
                break

    # Objective
    objective = ""
    obj_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if obj_match:
        objective = obj_match.group(1).strip()[:200]

    # Topics
    topics = extract_topics(content)

    return {
        "id": folder,
        "label": label,
        "name": name,
        "date": date_str,
        "type": session_type,
        "objective": objective,
        "topics": topics
    }


def generate_label(name: str, content: str) -> str:
    """Short label."""
    patterns = [
        (r'YOLO.*Resume', 'YOLO Resume'),
        (r'YOLO.*Fix', 'YOLO Fix'),
        (r'YOLO.*LR', 'YOLO LR'),
        (r'DETR.*EL2N|EL2N.*DETR', 'DETR EL2N'),
        (r'Query.*81', 'Query 81'),
        (r'IEEE.*Article', 'IEEE Article'),
        (r'IEEE.*Pipeline', 'IEEE Pipeline'),
        (r'IEEE.*Cleanup', 'IEEE Cleanup'),
        (r'IEEE.*Verif', 'IEEE Verify'),
        (r'Dataset.*Selection', 'Dataset Select'),
        (r'Scientific.*Justif', 'Justification'),
        (r'GPU.*Fix|PyTorch.*GPU', 'GPU Fix'),
        (r'Visualization', 'Visualization'),
    ]
    combined = f"{name} {content[:500]}"
    for pattern, lbl in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return lbl
    name_clean = re.sub(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_?', '', name).replace('_', ' ').strip()
    return name_clean[:20] if name_clean else "Session"


def extract_topics(content: str) -> list[str]:
    """Extract topics."""
    topics = set()
    patterns = {
        'DETR': r'\bDETR\b',
        'YOLO': r'\bYOLO\b',
        'DINO': r'\bDINO\b',
        'SAM': r'\bSAM\b',
        'EL2N': r'\bEL2N\b',
        'Fourier': r'\bFourier\b',
        'IEEE': r'\bIEEE\b',
        'Dataset': r'\bDataset\s*Selection',
        'Pipeline': r'\bPipeline\b',
        'SSH/Eden': r'\bSSH\b|\bEden\b',
        'GPU': r'\bGPU\b|\bCUDA\b',
        'Training': r'\bTraining\b',
    }
    for topic, pattern in patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            topics.add(topic)
    return sorted(topics)


def build_graph(sessions: list[dict]) -> tuple[list, list]:
    """Build nodes and edges for graph."""
    nodes = []
    edges = []

    # Type colors
    type_colors = {
        "Training": "#ff6b6b",
        "Analysis": "#4ecdc4",
        "Writing": "#ffeaa7",
        "Benchmark": "#a29bfe",
        "SSH": "#55efc4",
        "Mixed": "#b2bec3"
    }

    # Add session nodes
    for s in sessions:
        nodes.append({
            "id": s["id"],
            "label": s["label"],
            "title": f"<b>{s['label']}</b><br>{s['date']}<br>{s['type']}<br><br>{s['objective'][:100]}...",
            "group": "session",
            "color": type_colors.get(s["type"], "#b2bec3"),
            "shape": "dot",
            "size": 20,
            "font": {"size": 12, "color": "#ffffff"},
            "data": s
        })

    # Add topic nodes
    all_topics = set()
    for s in sessions:
        all_topics.update(s["topics"])

    topic_colors = {
        "DETR": "#e74c3c",
        "YOLO": "#2ecc71",
        "DINO": "#9b59b6",
        "IEEE": "#f39c12",
        "Dataset": "#3498db",
        "Pipeline": "#1abc9c",
        "Training": "#e91e63",
        "GPU": "#00bcd4",
        "SSH/Eden": "#8bc34a",
        "EL2N": "#ff5722",
        "Fourier": "#673ab7",
        "SAM": "#795548"
    }

    for topic in all_topics:
        nodes.append({
            "id": f"topic_{topic}",
            "label": topic,
            "title": f"<b>Topic: {topic}</b>",
            "group": "topic",
            "color": topic_colors.get(topic, "#95a5a6"),
            "shape": "diamond",
            "size": 15,
            "font": {"size": 11, "color": "#ffffff"}
        })

    # Add edges: session -> topics
    for s in sessions:
        for topic in s["topics"]:
            edges.append({
                "from": s["id"],
                "to": f"topic_{topic}",
                "color": {"color": "rgba(255,255,255,0.2)", "highlight": "#667eea"},
                "width": 1
            })

    # Add edges: session -> session (shared topics >= 3)
    for i, s1 in enumerate(sessions):
        for j, s2 in enumerate(sessions):
            if i >= j:
                continue
            shared = set(s1["topics"]) & set(s2["topics"])
            if len(shared) >= 4:
                edges.append({
                    "from": s1["id"],
                    "to": s2["id"],
                    "color": {"color": "rgba(102, 126, 234, 0.4)", "highlight": "#667eea"},
                    "width": 2,
                    "dashes": True,
                    "title": f"Shared: {', '.join(shared)}"
                })

    return nodes, edges


def generate_html(nodes: list, edges: list) -> str:
    """Generate HTML with vis.js graph."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Session Graph View</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #0f0f1a;
            color: #eee;
            overflow: hidden;
        }}
        #graph {{
            width: 100vw;
            height: 100vh;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(15, 15, 26, 0.95);
            padding: 10px 20px;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }}
        .header h1 {{ font-size: 1.2em; }}
        .controls {{
            display: flex;
            gap: 10px;
        }}
        .controls button {{
            padding: 6px 12px;
            background: #667eea;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-size: 0.85em;
        }}
        .controls button:hover {{ background: #5a6fd6; }}
        .legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(15, 15, 26, 0.95);
            padding: 15px;
            border-radius: 10px;
            font-size: 0.8em;
            z-index: 100;
        }}
        .legend-title {{ font-weight: 600; margin-bottom: 8px; color: #667eea; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .legend-diamond {{ width: 10px; height: 10px; transform: rotate(45deg); }}

        .detail-panel {{
            position: fixed;
            top: 50px;
            right: 0;
            width: 320px;
            height: calc(100vh - 50px);
            background: rgba(15, 15, 26, 0.98);
            border-left: 1px solid #333;
            padding: 20px;
            z-index: 100;
            overflow-y: auto;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }}
        .detail-panel.open {{ transform: translateX(0); }}
        .detail-panel h2 {{ font-size: 1.1em; margin-bottom: 10px; color: #667eea; }}
        .detail-panel .date {{ color: #888; font-size: 0.85em; margin-bottom: 15px; }}
        .detail-panel .type-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-bottom: 15px;
        }}
        .detail-panel .objective {{
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
            padding: 10px;
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 15px;
        }}
        .detail-panel .topics {{ margin-bottom: 15px; }}
        .detail-panel .topic-tag {{
            display: inline-block;
            padding: 3px 8px;
            background: rgba(102, 126, 234, 0.2);
            border-radius: 6px;
            font-size: 0.75em;
            margin: 2px;
            color: #a8b4ff;
        }}
        .detail-panel .path {{
            font-family: monospace;
            font-size: 0.75em;
            background: rgba(0,0,0,0.3);
            padding: 8px;
            border-radius: 4px;
            color: #666;
            word-break: break-all;
        }}
        .close-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            color: #666;
            font-size: 1.5em;
            cursor: pointer;
        }}
        .close-btn:hover {{ color: #fff; }}

        .stats {{
            font-size: 0.85em;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Session Graph View</h1>
        <div class="controls">
            <button onclick="network.fit()">Fit View</button>
            <button onclick="togglePhysics()">Toggle Physics</button>
            <button onclick="filterSessions()">Sessions Only</button>
            <button onclick="showAll()">Show All</button>
        </div>
        <div class="stats" id="stats"></div>
    </div>

    <div id="graph"></div>

    <div class="legend">
        <div class="legend-title">Session Types</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div> Training</div>
        <div class="legend-item"><div class="legend-dot" style="background:#4ecdc4"></div> Analysis</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ffeaa7"></div> Writing</div>
        <div class="legend-item"><div class="legend-dot" style="background:#a29bfe"></div> Benchmark</div>
        <div class="legend-item"><div class="legend-dot" style="background:#55efc4"></div> SSH</div>
        <div class="legend-item"><div class="legend-dot" style="background:#b2bec3"></div> Mixed</div>
        <div class="legend-title" style="margin-top:12px">Topics</div>
        <div class="legend-item"><div class="legend-diamond" style="background:#95a5a6"></div> Topic node</div>
    </div>

    <div class="detail-panel" id="detailPanel">
        <button class="close-btn" onclick="closePanel()">&times;</button>
        <div id="detailContent"></div>
    </div>

    <script>
    const nodesData = {json.dumps(nodes, ensure_ascii=False)};
    const edgesData = {json.dumps(edges, ensure_ascii=False)};

    const nodes = new vis.DataSet(nodesData);
    const edges = new vis.DataSet(edgesData);

    const container = document.getElementById('graph');
    const data = {{ nodes: nodes, edges: edges }};

    const options = {{
        physics: {{
            enabled: true,
            barnesHut: {{
                gravitationalConstant: -2000,
                centralGravity: 0.5,
                springLength: 95,
                springConstant: 0.05,
                damping: 0.2,
                avoidOverlap: 0.1
            }},
            stabilization: {{
                enabled: true,
                iterations: 40,
                updateInterval: 50,
                fit: true
            }},
            adaptiveTimestep: true,
            maxVelocity: 50,
            minVelocity: 1.0
        }},
        nodes: {{
            borderWidth: 2,
            shadow: false,
            font: {{
                color: '#ffffff'
            }}
        }},
        edges: {{
            smooth: {{
                enabled: true,
                type: 'dynamic',
                roundness: 0.5
            }},
            shadow: false
        }},
        interaction: {{
            hover: true,
            tooltipDelay: 100,
            zoomView: true,
            dragView: true
        }}
    }};

    const network = new vis.Network(container, data, options);

    // Auto-disable physics after stabilization (performance boost)
    network.once('stabilizationIterationsDone', function() {{
        network.setOptions({{ physics: {{ enabled: false }} }});
        physicsEnabled = false;
    }});

    // Stats
    const sessionCount = nodesData.filter(n => n.group === 'session').length;
    const topicCount = nodesData.filter(n => n.group === 'topic').length;
    document.getElementById('stats').textContent = `${{sessionCount}} sessions | ${{topicCount}} topics | ${{edgesData.length}} connections`;

    // Click handler
    network.on('click', function(params) {{
        if (params.nodes.length > 0) {{
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            if (node.data) {{
                showDetail(node.data);
            }}
        }}
    }});

    // Double click to focus
    network.on('doubleClick', function(params) {{
        if (params.nodes.length > 0) {{
            network.focus(params.nodes[0], {{
                scale: 1.5,
                animation: true
            }});
        }}
    }});

    function showDetail(session) {{
        const panel = document.getElementById('detailPanel');
        const content = document.getElementById('detailContent');

        const typeColors = {{
            "Training": "#ff6b6b",
            "Analysis": "#4ecdc4",
            "Writing": "#ffeaa7",
            "Benchmark": "#a29bfe",
            "SSH": "#55efc4",
            "Mixed": "#b2bec3"
        }};

        content.innerHTML = `
            <h2>${{session.label}}</h2>
            <div class="date">${{session.date}} | ${{session.name}}</div>
            <span class="type-badge" style="background:${{typeColors[session.type] || '#b2bec3'}}40; color:${{typeColors[session.type] || '#b2bec3'}}">${{session.type}}</span>
            <div class="objective">${{session.objective || 'No objective recorded'}}</div>
            <div class="topics">
                ${{session.topics.map(t => `<span class="topic-tag">${{t}}</span>`).join('')}}
            </div>
            <div class="path">.sessions/${{session.id}}/</div>
        `;

        panel.classList.add('open');
    }}

    function closePanel() {{
        document.getElementById('detailPanel').classList.remove('open');
    }}

    let physicsEnabled = true;
    function togglePhysics() {{
        physicsEnabled = !physicsEnabled;
        network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
    }}

    function filterSessions() {{
        const sessionNodes = nodesData.filter(n => n.group === 'session');
        const sessionIds = sessionNodes.map(n => n.id);
        const sessionEdges = edgesData.filter(e =>
            sessionIds.includes(e.from) && sessionIds.includes(e.to)
        );
        nodes.clear();
        edges.clear();
        nodes.add(sessionNodes);
        edges.add(sessionEdges);
    }}

    function showAll() {{
        nodes.clear();
        edges.clear();
        nodes.add(nodesData);
        edges.add(edgesData);
    }}
    </script>
</body>
</html>'''


def main():
    print("Loading sessions...")
    sessions = []
    for d in SESSIONS_PATH.iterdir():
        if d.is_dir() and d.name.startswith('Session_'):
            s = parse_session(d)
            if s:
                sessions.append(s)

    sessions.sort(key=lambda x: x['date'])
    print(f"Loaded {len(sessions)} sessions")

    print("Building graph...")
    nodes, edges = build_graph(sessions)
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

    print("Generating HTML...")
    html = generate_html(nodes, edges)

    output_path = SESSIONS_PATH / "analysis" / "graph_view.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
