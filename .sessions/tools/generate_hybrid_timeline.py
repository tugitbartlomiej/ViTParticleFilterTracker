"""
Generate Hybrid Timeline - vertical tree with connection lines between related sessions.
Like git graph but for sessions.
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
    """Parse session."""
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
    if date_match:
        date_str = f"{date_match.group(1)}T{date_match.group(2)[:2]}:{date_match.group(2)[2:4]}"
    else:
        date_str = "2025-01-01T00:00"

    # Name
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder)
    name = desc_match.group(1).replace('_', ' ') if desc_match else folder

    # Label
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

    # Semantic summary
    semantic = generate_semantic(content, name)

    # Actions, findings for details
    actions = []
    actions_match = re.search(r'## Actions Taken\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if actions_match:
        items = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\n##|\Z)', actions_match.group(1), re.DOTALL)
        actions = [re.sub(r'\s+', ' ', a.strip())[:100] for a in items[:5]]

    findings = []
    findings_match = re.search(r'### Key Findings\n(.+?)(?=\n###|\n##|\Z)', content, re.DOTALL)
    if findings_match:
        items = re.findall(r'[-*]\s+(.+?)(?:\n|$)', findings_match.group(1))
        findings = [f[:80] for f in items[:4]]

    files = re.findall(r'`([^`]+\.(?:py|md|tex|yaml|json))`', content)[:5]

    next_steps = []
    next_match = re.search(r'## Next Steps\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if next_match:
        items = re.findall(r'\[.\]\s+(.+?)(?:\n|$)', next_match.group(1))
        next_steps = [n[:60] for n in items[:3]]

    return {
        "id": folder,
        "name": name,
        "label": label,
        "date": date_str,
        "type": session_type,
        "objective": objective,
        "topics": topics,
        "semantic": semantic,
        "actions": actions,
        "findings": findings,
        "files": files,
        "next_steps": next_steps
    }


def generate_label(name: str, content: str) -> str:
    """Generate short label."""
    patterns = [
        (r'YOLO.*Resume|Resume.*YOLO', 'YOLO Resume'),
        (r'YOLO.*Fix|Fix.*YOLO', 'YOLO Fix'),
        (r'YOLO.*LR', 'YOLO LR'),
        (r'DETR.*EL2N|EL2N.*DETR', 'DETR EL2N'),
        (r'Query.*81', 'Query 81'),
        (r'IEEE.*Article', 'IEEE Article'),
        (r'IEEE.*Pipeline', 'IEEE Pipeline'),
        (r'IEEE.*Cleanup', 'IEEE Cleanup'),
        (r'IEEE.*Verif', 'IEEE Verify'),
        (r'Dataset.*Selection', 'Dataset Select'),
        (r'Scientific.*Justif', 'Justification'),
        (r'PyTorch.*GPU|GPU.*Fix', 'GPU Fix'),
        (r'Visualization', 'Visualization'),
        (r'Benchmark.*Fix', 'Bench Fix'),
    ]
    combined = f"{name} {content[:500]}"
    for pattern, lbl in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return lbl
    name_clean = re.sub(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_?', '', name).replace('_', ' ').strip()
    return name_clean[:25] if name_clean else "Session"


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
        'Dataset': r'\bDataset\b',
        'Pipeline': r'\bPipeline\b',
        'SSH': r'\bSSH\b|\bEden\b',
        'GPU': r'\bGPU\b',
        'Query81': r'Query\s*81',
    }
    for topic, pattern in patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            topics.add(topic)
    return sorted(topics)


def generate_semantic(content: str, name: str) -> str:
    """Generate semantic summary."""
    results = []

    findings_match = re.search(r'### Key Findings\n(.+?)(?=\n###|\n##|\Z)', content, re.DOTALL)
    if findings_match:
        items = re.findall(r'[-*]\s*\*\*(.+?)\*\*', findings_match.group(1))
        if items:
            results.extend(items[:2])

    if not results:
        obj_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if obj_match:
            obj = re.sub(r'\s+', ' ', obj_match.group(1).strip())
            return obj[:120]

    if results:
        return '. '.join(results[:2])[:120]

    return name[:80]


def find_connections(sessions: list[dict]) -> list[dict]:
    """Find connections between sessions based on shared topics and proximity."""
    connections = []

    for i, s1 in enumerate(sessions):
        for j, s2 in enumerate(sessions):
            if i >= j:
                continue

            shared = set(s1["topics"]) & set(s2["topics"])

            # Strong connection: 3+ shared topics
            if len(shared) >= 3:
                connections.append({
                    "from": i,
                    "to": j,
                    "strength": "strong",
                    "topics": list(shared)
                })
            # Medium: 2 shared topics and close in time
            elif len(shared) >= 2 and abs(i - j) <= 5:
                connections.append({
                    "from": i,
                    "to": j,
                    "strength": "medium",
                    "topics": list(shared)
                })

    return connections


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

    # Find connections
    print("Finding connections...")
    connections = find_connections(sessions)
    print(f"Found {len(connections)} connections")

    # Assign indices
    for i, s in enumerate(sessions):
        s['index'] = i

    # Load HTML template
    template_path = SESSIONS_PATH / "analysis" / "hybrid_timeline_template.html"
    if not template_path.exists():
        # Create template
        create_template(template_path)

    html = template_path.read_text(encoding='utf-8')

    # Insert data
    html = html.replace('SESSION_DATA_PLACEHOLDER', json.dumps(sessions, indent=2, ensure_ascii=False))
    html = html.replace('CONNECTION_DATA_PLACEHOLDER', json.dumps(connections, indent=2, ensure_ascii=False))

    # Save
    output_path = SESSIONS_PATH / "analysis" / "hybrid_timeline.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"\nSaved to: {output_path}")


def create_template(path: Path):
    """Create HTML template."""
    template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hybrid Timeline</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: #eee;
        }
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(15, 15, 26, 0.98);
            padding: 12px 20px;
            z-index: 100;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.2em; }
        .header .stats { font-size: 0.85em; color: #888; }
        .legend {
            display: flex;
            gap: 15px;
            font-size: 0.8em;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .container {
            display: flex;
            padding-top: 60px;
        }

        /* Timeline */
        .timeline {
            flex: 1;
            padding: 20px 20px 40px 80px;
            max-width: 700px;
            margin: 0 auto;
            position: relative;
            z-index: 2;
        }
        .timeline::before {
            content: '';
            position: absolute;
            left: 44px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to top, #667eea, #764ba2);
        }

        .month-group { margin-bottom: 25px; }
        .month-label {
            position: relative;
            left: -50px;
            font-size: 0.7em;
            color: #667eea;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .session {
            position: relative;
            margin-bottom: 10px;
            padding: 10px 14px;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            border-left: 3px solid;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .session:hover {
            background: rgba(255,255,255,0.1);
            transform: translateX(5px);
        }
        .session.highlighted {
            background: rgba(102, 126, 234, 0.2);
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
        }
        .session::before {
            content: '';
            position: absolute;
            left: -39px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 3px solid;
            background: #0f0f1a;
        }

        .session.Training { border-color: #ff6b6b; }
        .session.Training::before { border-color: #ff6b6b; }
        .session.Analysis { border-color: #4ecdc4; }
        .session.Analysis::before { border-color: #4ecdc4; }
        .session.Writing { border-color: #ffeaa7; }
        .session.Writing::before { border-color: #ffeaa7; }
        .session.Benchmark { border-color: #a29bfe; }
        .session.Benchmark::before { border-color: #a29bfe; }
        .session.SSH { border-color: #55efc4; }
        .session.SSH::before { border-color: #55efc4; }
        .session.Mixed { border-color: #b2bec3; }
        .session.Mixed::before { border-color: #b2bec3; }

        .session-title {
            font-size: 1em;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .session-type {
            font-size: 0.65em;
            padding: 2px 6px;
            border-radius: 8px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .session.Training .session-type { background: rgba(255,107,107,0.3); color: #ff6b6b; }
        .session.Analysis .session-type { background: rgba(78,205,196,0.3); color: #4ecdc4; }
        .session.Writing .session-type { background: rgba(255,234,167,0.3); color: #ffeaa7; }
        .session.Benchmark .session-type { background: rgba(162,155,254,0.3); color: #a29bfe; }
        .session.SSH .session-type { background: rgba(85,239,196,0.3); color: #55efc4; }
        .session.Mixed .session-type { background: rgba(178,190,195,0.3); color: #b2bec3; }

        .session-meta {
            font-size: 0.75em;
            color: #888;
            margin-top: 3px;
        }
        .session-topics {
            margin-top: 5px;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }
        .topic-tag {
            font-size: 0.65em;
            padding: 2px 6px;
            background: rgba(102, 126, 234, 0.2);
            border-radius: 6px;
            color: #a8b4ff;
        }

        .session-details {
            display: none;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255,255,255,0.1);
            animation: slideDown 0.2s ease;
        }
        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .session.expanded .session-details { display: block; }

        .details-section { margin-bottom: 8px; }
        .details-title {
            font-size: 0.7em;
            color: #667eea;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 3px;
        }
        .details-list {
            font-size: 0.8em;
            color: #aaa;
            padding-left: 15px;
            margin: 0;
        }
        .details-list li { margin: 2px 0; }
        .details-objective {
            background: rgba(102, 126, 234, 0.1);
            border-left: 2px solid #667eea;
            padding: 6px 10px;
            font-size: 0.8em;
            color: #ccc;
            margin-bottom: 8px;
        }
        .details-path {
            font-family: monospace;
            font-size: 0.7em;
            background: rgba(0,0,0,0.3);
            padding: 4px 8px;
            border-radius: 4px;
            color: #666;
        }

        /* Tooltip */
        .tooltip {
            position: fixed;
            background: rgba(20, 20, 35, 0.98);
            border: 1px solid #667eea;
            border-radius: 8px;
            padding: 10px 14px;
            max-width: 300px;
            font-size: 0.85em;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }
        .tooltip.visible { opacity: 1; }
        .tooltip-label { font-weight: 600; color: #fff; margin-bottom: 4px; }
        .tooltip-semantic { color: #a8e6cf; font-style: italic; font-size: 0.9em; }

        /* Side panel for connections */
        .side-panel {
            position: fixed;
            right: 0;
            top: 60px;
            width: 280px;
            height: calc(100vh - 60px);
            background: rgba(15, 15, 26, 0.95);
            border-left: 1px solid #333;
            padding: 15px;
            overflow-y: auto;
            z-index: 50;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        .side-panel.open { transform: translateX(0); }
        .side-panel h3 {
            font-size: 0.9em;
            color: #667eea;
            margin-bottom: 10px;
        }
        .connection-item {
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 0.8em;
            cursor: pointer;
        }
        .connection-item:hover {
            background: rgba(102, 126, 234, 0.2);
        }
        .shared-topics {
            color: #888;
            font-size: 0.85em;
            margin-top: 3px;
        }

        .toggle-panel {
            position: fixed;
            right: 10px;
            bottom: 20px;
            padding: 10px 15px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85em;
            z-index: 60;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Hybrid Timeline</h1>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div>Training</div>
            <div class="legend-item"><div class="legend-dot" style="background:#4ecdc4"></div>Analysis</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ffeaa7"></div>Writing</div>
            <div class="legend-item"><div class="legend-dot" style="background:#a29bfe"></div>Benchmark</div>
            <div class="legend-item"><div class="legend-dot" style="background:#55efc4"></div>SSH</div>
            <div class="legend-item"><div class="legend-dot" style="background:#b2bec3"></div>Mixed</div>
        </div>
        <div class="stats" id="stats"></div>
    </div>

    <div class="container">
        <div class="timeline" id="timeline"></div>
    </div>

    <div class="side-panel" id="sidePanel">
        <h3>Related Sessions</h3>
        <div id="relatedList"></div>
    </div>

    <button class="toggle-panel" id="togglePanel">Show Connections</button>

    <div class="tooltip" id="tooltip">
        <div class="tooltip-label" id="tooltipLabel"></div>
        <div class="tooltip-semantic" id="tooltipSemantic"></div>
    </div>

    <script>
    const sessions = SESSION_DATA_PLACEHOLDER;
    const connections = CONNECTION_DATA_PLACEHOLDER;

    const sessionElements = [];

    // Group by month
    const byMonth = {};
    sessions.forEach((s, i) => {
        s.index = i;
        const month = s.date.substring(0, 7);
        if (!byMonth[month]) byMonth[month] = [];
        byMonth[month].push(s);
    });

    const sortedMonths = Object.keys(byMonth).sort();
    const timeline = document.getElementById('timeline');

    sortedMonths.forEach(month => {
        const group = document.createElement('div');
        group.className = 'month-group';

        const label = document.createElement('div');
        label.className = 'month-label';
        label.textContent = month;
        group.appendChild(label);

        byMonth[month].sort((a, b) => a.date.localeCompare(b.date));

        byMonth[month].forEach(s => {
            const el = document.createElement('div');
            el.className = `session ${s.type}`;
            el.dataset.index = s.index;
            el.dataset.label = s.label;
            el.dataset.semantic = s.semantic || '';

            const topicTags = s.topics.slice(0, 4).map(t => `<span class="topic-tag">${t}</span>`).join('');

            let detailsHtml = '';
            if (s.objective) detailsHtml += `<div class="details-objective">${s.objective}</div>`;
            if (s.actions && s.actions.length) {
                detailsHtml += `<div class="details-section"><div class="details-title">Actions</div><ul class="details-list">${s.actions.map(a => `<li>${a}</li>`).join('')}</ul></div>`;
            }
            if (s.findings && s.findings.length) {
                detailsHtml += `<div class="details-section"><div class="details-title">Findings</div><ul class="details-list">${s.findings.map(f => `<li>${f}</li>`).join('')}</ul></div>`;
            }
            if (s.files && s.files.length) {
                detailsHtml += `<div class="details-section"><div class="details-title">Files</div><ul class="details-list">${s.files.map(f => `<li><code>${f}</code></li>`).join('')}</ul></div>`;
            }
            detailsHtml += `<div class="details-path">.sessions/${s.id}/</div>`;

            el.innerHTML = `
                <div class="session-title">
                    ${s.label}
                    <span class="session-type">${s.type}</span>
                </div>
                <div class="session-meta">${s.date.substring(5, 10)} | ${s.name}</div>
                <div class="session-topics">${topicTags}</div>
                <div class="session-details">${detailsHtml}</div>
            `;

            el.addEventListener('click', () => {
                el.classList.toggle('expanded');
                showRelated(s.index);
            });

            group.appendChild(el);
            sessionElements[s.index] = el;
        });

        timeline.appendChild(group);
    });

    // Stats
    document.getElementById('stats').textContent = `${sessions.length} sessions | ${connections.length} connections`;

    // Tooltip
    const tooltip = document.getElementById('tooltip');
    const tooltipLabel = document.getElementById('tooltipLabel');
    const tooltipSemantic = document.getElementById('tooltipSemantic');

    document.querySelectorAll('.session').forEach(el => {
        el.addEventListener('mouseenter', () => {
            tooltipLabel.textContent = el.dataset.label;
            tooltipSemantic.textContent = el.dataset.semantic;
            tooltip.classList.add('visible');
        });
        el.addEventListener('mousemove', (e) => {
            tooltip.style.left = Math.min(e.clientX + 15, window.innerWidth - 320) + 'px';
            tooltip.style.top = Math.min(e.clientY + 15, window.innerHeight - 100) + 'px';
        });
        el.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
    });

    // Side panel
    const sidePanel = document.getElementById('sidePanel');
    const relatedList = document.getElementById('relatedList');
    const toggleBtn = document.getElementById('togglePanel');

    toggleBtn.addEventListener('click', () => {
        sidePanel.classList.toggle('open');
        toggleBtn.textContent = sidePanel.classList.contains('open') ? 'Hide Connections' : 'Show Connections';
    });

    function showRelated(index) {
        const related = connections.filter(c => c.from === index || c.to === index);

        // Clear highlights
        document.querySelectorAll('.session.highlighted').forEach(el => el.classList.remove('highlighted'));

        // Highlight related
        related.forEach(c => {
            const otherIndex = c.from === index ? c.to : c.from;
            if (sessionElements[otherIndex]) {
                sessionElements[otherIndex].classList.add('highlighted');
            }
        });

        // Update side panel
        relatedList.innerHTML = related.map(c => {
            const otherIndex = c.from === index ? c.to : c.from;
            const other = sessions[otherIndex];
            return `
                <div class="connection-item" data-index="${otherIndex}">
                    <strong>${other.label}</strong> (${other.date.substring(5, 10)})
                    <div class="shared-topics">Shared: ${c.topics.join(', ')}</div>
                </div>
            `;
        }).join('') || '<div style="color:#666">No connections found</div>';

        // Click to scroll
        relatedList.querySelectorAll('.connection-item').forEach(item => {
            item.addEventListener('click', () => {
                const idx = parseInt(item.dataset.index);
                sessionElements[idx]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                sessionElements[idx]?.classList.add('expanded');
            });
        });

        if (related.length > 0) {
            sidePanel.classList.add('open');
            toggleBtn.textContent = 'Hide Connections';
        }

    }

    // Scroll to newest
    window.scrollTo(0, document.body.scrollHeight);
    </script>
</body>
</html>'''
    path.write_text(template, encoding='utf-8')
    print(f"Created template: {path}")


if __name__ == "__main__":
    main()
