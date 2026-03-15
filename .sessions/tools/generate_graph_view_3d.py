"""
Generate interactive 3D graph view using 3d-force-graph library.
Pure HTML/JS, no install needed - uses CDN.
"""

import io
import re
import json
import sys
from pathlib import Path

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
        (r'RAG', 'RAG'),
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


def build_graph(sessions: list[dict]) -> dict:
    """Build nodes and links for 3d-force-graph."""
    nodes = []
    links = []

    # Type colors - more vibrant
    type_colors = {
        "Training": "#ff4757",
        "Analysis": "#1dd1a1",
        "Writing": "#feca57",
        "Benchmark": "#a55eea",
        "SSH": "#00d2d3",
        "Mixed": "#8395a7"
    }

    topic_colors = {
        "DETR": "#ff6b6b",
        "YOLO": "#26de81",
        "DINO": "#a55eea",
        "IEEE": "#fed330",
        "Dataset": "#45aaf2",
        "Pipeline": "#2bcbba",
        "Training": "#fc5c65",
        "GPU": "#4bcffa",
        "SSH/Eden": "#20bf6b",
        "EL2N": "#fd9644",
        "Fourier": "#8854d0",
        "SAM": "#a5673f"
    }

    # Add session nodes
    for s in sessions:
        nodes.append({
            "id": s["id"],
            "label": s["label"],
            "group": "session",
            "color": type_colors.get(s["type"], "#b2bec3"),
            "size": 8,
            "data": s
        })

    # Add topic nodes
    all_topics = set()
    for s in sessions:
        all_topics.update(s["topics"])

    for topic in all_topics:
        nodes.append({
            "id": f"topic_{topic}",
            "label": topic,
            "group": "topic",
            "color": topic_colors.get(topic, "#95a5a6"),
            "size": 5
        })

    # Add links: session -> topics
    for s in sessions:
        for topic in s["topics"]:
            topic_color = topic_colors.get(topic, "#95a5a6")
            links.append({
                "source": s["id"],
                "target": f"topic_{topic}",
                "color": f"{topic_color}60",  # 60 = ~38% opacity in hex
                "width": 1
            })

    # Add links: session -> session (shared topics >= 2)
    for i, s1 in enumerate(sessions):
        for j, s2 in enumerate(sessions):
            if i >= j:
                continue
            shared = set(s1["topics"]) & set(s2["topics"])
            if len(shared) >= 2:
                # Stronger link for more shared topics
                strength = min(len(shared), 5)
                links.append({
                    "source": s1["id"],
                    "target": s2["id"],
                    "color": f"rgba(102, 126, 234, {0.3 + strength * 0.1})",
                    "width": 1 + strength * 0.5
                })

    return {"nodes": nodes, "links": links}


def generate_html(graph_data: dict) -> str:
    """Generate HTML with 3d-force-graph."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Session Graph 3D</title>
    <script src="https://unpkg.com/3d-force-graph@1"></script>
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three-spritetext@1"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #050510;
            color: #eee;
            overflow: hidden;
        }}
        #graph {{ width: 100vw; height: 100vh; }}
        
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(10, 10, 20, 0.95);
            padding: 10px 20px;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }}
        .header h1 {{ font-size: 1.2em; }}
        .controls {{ display: flex; gap: 10px; }}
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
        .stats {{ font-size: 0.85em; color: #888; }}
        
        .legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(10, 10, 20, 0.95);
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
            background: rgba(10, 10, 20, 0.98);
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
        
        .help {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(10, 10, 20, 0.95);
            padding: 12px;
            border-radius: 10px;
            font-size: 0.75em;
            color: #666;
            z-index: 100;
        }}
        .help div {{ margin: 3px 0; }}
        .help kbd {{
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Session Graph 3D</h1>
        <div class="controls">
            <button onclick="resetCamera()">Reset View</button>
            <button onclick="toggleRotation()">Auto Rotate</button>
            <button onclick="filterSessions()">Sessions Only</button>
            <button onclick="showAll()">Show All</button>
        </div>
        <div class="stats" id="stats"></div>
    </div>

    <div id="graph"></div>

    <div class="legend">
        <div class="legend-title">Session Types</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff4757;box-shadow:0 0 8px #ff4757"></div> Training</div>
        <div class="legend-item"><div class="legend-dot" style="background:#1dd1a1;box-shadow:0 0 8px #1dd1a1"></div> Analysis</div>
        <div class="legend-item"><div class="legend-dot" style="background:#feca57;box-shadow:0 0 8px #feca57"></div> Writing</div>
        <div class="legend-item"><div class="legend-dot" style="background:#a55eea;box-shadow:0 0 8px #a55eea"></div> Benchmark</div>
        <div class="legend-item"><div class="legend-dot" style="background:#00d2d3;box-shadow:0 0 8px #00d2d3"></div> SSH</div>
        <div class="legend-item"><div class="legend-dot" style="background:#8395a7;box-shadow:0 0 8px #8395a7"></div> Mixed</div>
        <div class="legend-title" style="margin-top:12px">Topics</div>
        <div class="legend-item"><div class="legend-diamond" style="background:#667eea;box-shadow:0 0 6px #667eea"></div> Topic node (smaller)</div>
    </div>

    <div class="detail-panel" id="detailPanel">
        <button class="close-btn" onclick="closePanel()">&times;</button>
        <div id="detailContent"></div>
    </div>
    
    <div class="help">
        <div><kbd>Drag</kbd> Rotate view</div>
        <div><kbd>Scroll</kbd> Zoom</div>
        <div><kbd>Click</kbd> Select node</div>
        <div><kbd>Double-click</kbd> Focus node</div>
    </div>

    <script>
    const graphData = {json.dumps(graph_data, ensure_ascii=False)};
    const allNodes = [...graphData.nodes];
    const allLinks = [...graphData.links];
    
    let isFiltered = false;
    let autoRotate = false;
    let focusedNode = null;

    // Highlight sets (built-in pattern from 3d-force-graph)
    const highlightNodes = new Set();
    const highlightLinks = new Set();
    let hoverNode = null;

    // Build neighbor map
    const neighbors = new Map();
    allLinks.forEach(link => {{
        const sid = link.source;
        const tid = link.target;
        if (!neighbors.has(sid)) neighbors.set(sid, new Set());
        if (!neighbors.has(tid)) neighbors.set(tid, new Set());
        neighbors.get(sid).add(tid);
        neighbors.get(tid).add(sid);
    }});

    const Graph = ForceGraph3D()
        (document.getElementById('graph'))
        .graphData(graphData)
        .backgroundColor('#050510')
        .showNavInfo(false)
        .nodeLabel(node => {{
            if (node.data) {{
                return `<div style="background:rgba(0,0,0,0.95);padding:12px 15px;border-radius:8px;max-width:300px;border:1px solid ${{node.color}}50;">
                    <b style="color:${{node.color}};font-size:1.1em">${{node.label}}</b><br>
                    <span style="color:#999;font-size:0.9em">${{node.data.date}} | ${{node.data.type}}</span><br>
                    <span style="color:#bbb;font-size:0.85em;line-height:1.4">${{node.data.objective?.substring(0,120) || ''}}...</span>
                </div>`;
            }}
            return `<b style="color:${{node.color}};font-size:1.1em">${{node.label}}</b>`;
        }})
        .nodeThreeObject(node => {{
            const group = new THREE.Group();
            const radius = node.group === 'session' ? 6 : 4;

            // Sphere
            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(radius, 32, 32),
                new THREE.MeshBasicMaterial({{ color: node.color, transparent: true, opacity: 1 }})
            );
            sphere.name = 'sphere';
            group.add(sphere);

            // Glow
            const glow = new THREE.Mesh(
                new THREE.SphereGeometry(radius * 1.5, 32, 32),
                new THREE.MeshBasicMaterial({{ color: node.color, transparent: true, opacity: 0.2 }})
            );
            glow.name = 'glow';
            group.add(glow);

            // Label below
            const label = new SpriteText(node.label);
            label.color = node.group === 'session' ? '#e0e0e0' : node.color;
            label.textHeight = node.group === 'session' ? 3.5 : 2.8;
            label.position.y = -(radius + 5);
            label.name = 'label';
            label.material.transparent = true;
            group.add(label);

            return group;
        }})
        .nodeThreeObjectExtend(false)
        .nodeVal(node => node.group === 'session' ? 12 : 6)
        .linkWidth(link => highlightLinks.has(link) ? 3 : 1)
        .linkColor(link => highlightLinks.has(link) ? '#88aaff' : 'rgba(100,130,255,0.15)')
        .linkDirectionalParticles(link => highlightLinks.has(link) ? 4 : 0)
        .linkDirectionalParticleWidth(2)
        .linkDirectionalParticleSpeed(0.006)
        .linkDirectionalParticleColor(() => '#88aaff')
        .onNodeHover(node => {{
            // Clear previous
            highlightNodes.clear();
            highlightLinks.clear();

            if (node) {{
                highlightNodes.add(node);
                const nodeNeighbors = neighbors.get(node.id);
                if (nodeNeighbors) {{
                    nodeNeighbors.forEach(id => {{
                        const n = allNodes.find(x => x.id === id);
                        if (n) highlightNodes.add(n);
                    }});
                }}
                allLinks.forEach(link => {{
                    const sid = typeof link.source === 'object' ? link.source.id : link.source;
                    const tid = typeof link.target === 'object' ? link.target.id : link.target;
                    if (sid === node.id || tid === node.id) highlightLinks.add(link);
                }});
            }}

            hoverNode = node || null;
            updateHighlight();
        }})
        .d3AlphaDecay(0.01)
        .d3VelocityDecay(0.3)
        .warmupTicks(100)
        .cooldownTicks(200)
        .onNodeClick(node => {{
            if (node.data) {{
                showDetail(node.data);
            }}
        }})
        .onNodeDblClick(node => {{
            // Focus on node and show only its connections
            focusOnNode(node);
        }})
        .onBackgroundClick(() => {{
            if (focusedNode) {{
                showAll();
                focusedNode = null;
            }}
            closePanel();
        }});

    // Add ambient light for better visibility
    Graph.scene().add(new THREE.AmbientLight(0xffffff, 0.8));
    Graph.scene().add(new THREE.DirectionalLight(0xffffff, 0.4));

    // Highlight function (Obsidian-style dimming)
    function updateHighlight() {{
        Graph.graphData().nodes.forEach(node => {{
            const obj = node.__threeObj;
            if (!obj) return;

            const isHl = !hoverNode || highlightNodes.has(node);
            obj.children.forEach(child => {{
                if (child.material) {{
                    if (child.name === 'sphere') {{
                        child.material.opacity = isHl ? 1.0 : 0.12;
                    }} else if (child.name === 'glow') {{
                        child.material.opacity = isHl ? 0.35 : 0.03;
                    }} else if (child.name === 'label') {{
                        child.material.opacity = isHl ? 1.0 : 0.1;
                    }}
                }}
            }});
        }});

        // Trigger link re-render
        Graph.linkWidth(Graph.linkWidth())
             .linkColor(Graph.linkColor())
             .linkDirectionalParticles(Graph.linkDirectionalParticles());
    }}
    
    // Update stats
    const sessionCount = allNodes.filter(n => n.group === 'session').length;
    const topicCount = allNodes.filter(n => n.group === 'topic').length;
    document.getElementById('stats').textContent = `${{sessionCount}} sessions | ${{topicCount}} topics | ${{allLinks.length}} connections`;
    
    function focusOnNode(node) {{
        focusedNode = node;
        
        // Find connected nodes
        const connectedIds = new Set([node.id]);
        allLinks.forEach(link => {{
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            if (sourceId === node.id) connectedIds.add(targetId);
            if (targetId === node.id) connectedIds.add(sourceId);
        }});
        
        // Filter graph
        const filteredNodes = allNodes.filter(n => connectedIds.has(n.id));
        const filteredLinks = allLinks.filter(link => {{
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return connectedIds.has(sourceId) && connectedIds.has(targetId);
        }});
        
        Graph.graphData({{ nodes: filteredNodes, links: filteredLinks }});
        isFiltered = true;
        
        // Camera focus
        const distance = 150;
        const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);
        Graph.cameraPosition(
            {{ x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio }},
            node,
            1000
        );
        
        document.getElementById('stats').textContent = `Showing: ${{filteredNodes.length}} nodes | ${{filteredLinks.length}} connections (click background to reset)`;
    }}
    
    function showDetail(session) {{
        const panel = document.getElementById('detailPanel');
        const content = document.getElementById('detailContent');

        const typeColors = {{
            "Training": "#ff4757",
            "Analysis": "#1dd1a1",
            "Writing": "#feca57",
            "Benchmark": "#a55eea",
            "SSH": "#00d2d3",
            "Mixed": "#8395a7"
        }};
        
        content.innerHTML = `
            <h2>${{session.label}}</h2>
            <div class="date">${{session.date}} | ${{session.name}}</div>
            <span class="type-badge" style="background:${{typeColors[session.type] || '#b2bec3'}}40; color:${{typeColors[session.type] || '#b2bec3'}}">${{session.type}}</span>
            <div class="objective">${{session.objective || 'No objective recorded'}}</div>
            <div class="topics">
                ${{session.topics.map(t => `<span class="topic-tag">${{t}}</span>`).join('')}}
            </div>
        `;
        
        panel.classList.add('open');
    }}
    
    function closePanel() {{
        document.getElementById('detailPanel').classList.remove('open');
    }}
    
    function resetCamera() {{
        Graph.cameraPosition({{ x: 200, y: 150, z: 400 }}, {{ x: 0, y: 0, z: 0 }}, 1500);
    }}

    // Set initial camera position
    setTimeout(() => {{
        Graph.cameraPosition({{ x: 200, y: 150, z: 400 }}, {{ x: 0, y: 0, z: 0 }}, 2000);
    }}, 500);

    function toggleRotation() {{
        autoRotate = !autoRotate;
        if (autoRotate) {{
            let angle = 0;
            (function rotate() {{
                if (!autoRotate) return;
                Graph.cameraPosition({{
                    x: 400 * Math.sin(angle),
                    y: 150,
                    z: 400 * Math.cos(angle)
                }});
                angle += Math.PI / 500;
                requestAnimationFrame(rotate);
            }})();
        }}
    }}
    
    function filterSessions() {{
        const sessionNodes = allNodes.filter(n => n.group === 'session');
        const sessionIds = new Set(sessionNodes.map(n => n.id));
        const sessionLinks = allLinks.filter(link => {{
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return sessionIds.has(sourceId) && sessionIds.has(targetId);
        }});
        Graph.graphData({{ nodes: sessionNodes, links: sessionLinks }});
        isFiltered = true;
        
        document.getElementById('stats').textContent = `${{sessionNodes.length}} sessions | ${{sessionLinks.length}} connections`;
    }}
    
    function showAll() {{
        Graph.graphData({{ nodes: allNodes, links: allLinks }});
        isFiltered = false;
        focusedNode = null;
        
        const sessionCount = allNodes.filter(n => n.group === 'session').length;
        const topicCount = allNodes.filter(n => n.group === 'topic').length;
        document.getElementById('stats').textContent = `${{sessionCount}} sessions | ${{topicCount}} topics | ${{allLinks.length}} connections`;
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
    graph_data = build_graph(sessions)
    print(f"Nodes: {len(graph_data['nodes'])}, Links: {len(graph_data['links'])}")

    print("Generating HTML...")
    html = generate_html(graph_data)

    output_path = SESSIONS_PATH / "analysis" / "graph_view_3d.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
