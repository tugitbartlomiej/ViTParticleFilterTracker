"""
Export sessions to Obsidian vault with [[wiki links]].
Creates interconnected notes for knowledge building.
"""

import io
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SESSIONS_PATH = Path(__file__).parent.parent
OUTPUT_PATH = SESSIONS_PATH / "obsidian_vault"


def parse_session(session_dir: Path) -> dict | None:
    """Parse session for Obsidian export."""
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
        date_str = date_match.group(1)
        time_str = f"{date_match.group(2)[:2]}:{date_match.group(2)[2:4]}"
    else:
        date_str = "Unknown"
        time_str = "00:00"

    # Name
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder)
    name = desc_match.group(1).replace('_', ' ') if desc_match else folder

    # Short label
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
        objective = obj_match.group(1).strip()

    # Topics
    topics = extract_topics(content)

    # Actions
    actions = []
    actions_match = re.search(r'## Actions Taken\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if actions_match:
        items = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\n##|\Z)', actions_match.group(1), re.DOTALL)
        actions = [re.sub(r'\s+', ' ', a.strip())[:200] for a in items[:10]]

    # Key findings
    findings = []
    findings_match = re.search(r'### Key Findings\n(.+?)(?=\n###|\n##|\Z)', content, re.DOTALL)
    if findings_match:
        items = re.findall(r'[-*]\s+(.+?)(?:\n|$)', findings_match.group(1))
        findings = [f.strip()[:150] for f in items[:6]]

    # Files
    files = re.findall(r'`([^`]+\.(?:py|md|tex|yaml|json|html))`', content)[:10]

    # Next steps
    next_steps = []
    next_match = re.search(r'## Next Steps\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if next_match:
        items = re.findall(r'\[.\]\s+(.+?)(?:\n|$)', next_match.group(1))
        next_steps = [n.strip()[:100] for n in items[:5]]

    # Related sessions mentioned
    related = re.findall(r'Session_\d{4}-\d{2}-\d{2}_\d{6}(?:_[\w]+)?', content)
    related = [r for r in related if r != folder]

    return {
        "id": folder,
        "name": name,
        "label": label,
        "date": date_str,
        "time": time_str,
        "type": session_type,
        "objective": objective,
        "topics": topics,
        "actions": actions,
        "findings": findings,
        "files": files,
        "next_steps": next_steps,
        "related": related,
        "content": content
    }


def generate_label(name: str, content: str) -> str:
    """Generate short label."""
    patterns = [
        (r'YOLO.*Resume|Resume.*YOLO', 'YOLO Resume'),
        (r'YOLO.*Fix|Fix.*YOLO', 'YOLO Fix'),
        (r'YOLO.*LR', 'YOLO LR'),
        (r'DETR.*EL2N|EL2N.*DETR', 'DETR EL2N'),
        (r'DETR.*Query|Query.*81', 'Query 81'),
        (r'IEEE.*Article|Article.*IEEE', 'IEEE Article'),
        (r'IEEE.*Pipeline', 'IEEE Pipeline'),
        (r'IEEE.*Cleanup', 'IEEE Cleanup'),
        (r'IEEE.*Verification', 'IEEE Verify'),
        (r'Dataset.*Selection', 'Dataset Selection'),
        (r'Scientific.*Justification', 'Justification'),
        (r'PyTorch.*GPU|GPU.*Fix', 'GPU Fix'),
        (r'Visualization|Visual', 'Visualization'),
    ]
    combined = f"{name} {content[:500]}"
    for pattern, label in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return label

    name_clean = re.sub(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_?', '', name).replace('_', ' ').strip()
    return name_clean[:30] if name_clean else "Session"


def extract_topics(content: str) -> list[str]:
    """Extract topics from content."""
    topics = set()
    patterns = {
        'DETR': r'\bDETR\b',
        'YOLO': r'\bYOLO\b',
        'DINO': r'\bDINO\b',
        'SAM': r'\bSAM\b|FastSAM',
        'EL2N': r'\bEL2N\b',
        'Fourier': r'\bFourier\b',
        'K-Means': r'\bK-Means\b|KMeans',
        'K-Center': r'\bK-Center\b',
        'IEEE': r'\bIEEE\b',
        'Dataset Selection': r'Dataset\s*Selection',
        'Pipeline': r'\bPipeline\b',
        'Training': r'\bTraining\b|train',
        'Benchmark': r'\bBenchmark\b',
        'SSH Eden': r'\bSSH\b|\bEden\b|\bSlurm\b',
        'GPU': r'\bGPU\b|\bCUDA\b',
        'Query 81': r'Query\s*81|Q81',
    }
    for topic, pattern in patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            topics.add(topic)
    return sorted(topics)


def find_related_by_topics(sessions: list[dict]) -> dict[str, list[str]]:
    """Find related sessions by shared topics."""
    relations = defaultdict(set)

    for i, s1 in enumerate(sessions):
        for j, s2 in enumerate(sessions):
            if i >= j:
                continue
            # Count shared topics
            shared = set(s1["topics"]) & set(s2["topics"])
            if len(shared) >= 2:  # At least 2 shared topics
                relations[s1["id"]].add(s2["id"])
                relations[s2["id"]].add(s1["id"])

    return {k: sorted(v) for k, v in relations.items()}


def create_session_note(session: dict, relations: dict[str, list[str]], all_sessions: dict[str, dict]) -> str:
    """Create Obsidian note for a session."""
    s = session

    # YAML frontmatter
    note = f"""---
date: {s['date']}
time: {s['time']}
type: {s['type']}
topics: [{', '.join(s['topics'])}]
aliases: ["{s['label']}"]
---

# {s['label']}

> [!info] Session Info
> **Date:** {s['date']} {s['time']}
> **Type:** {s['type']}
> **ID:** `{s['id']}`

## Objective

{s['objective'] if s['objective'] else '_No objective recorded_'}

"""

    # Topics as links
    if s['topics']:
        topic_links = ' '.join([f"[[{t}]]" for t in s['topics']])
        note += f"## Topics\n\n{topic_links}\n\n"

    # Actions
    if s['actions']:
        note += "## Actions Taken\n\n"
        for i, a in enumerate(s['actions'], 1):
            note += f"{i}. {a}\n"
        note += "\n"

    # Findings
    if s['findings']:
        note += "## Key Findings\n\n"
        for f in s['findings']:
            note += f"- {f}\n"
        note += "\n"

    # Files
    if s['files']:
        note += "## Files Modified\n\n"
        for f in s['files']:
            note += f"- `{f}`\n"
        note += "\n"

    # Next steps
    if s['next_steps']:
        note += "## Next Steps\n\n"
        for n in s['next_steps']:
            note += f"- [ ] {n}\n"
        note += "\n"

    # Related sessions
    related_ids = set(s.get('related', [])) | set(relations.get(s['id'], []))
    if related_ids:
        note += "## Related Sessions\n\n"
        for rid in sorted(related_ids):
            if rid in all_sessions:
                rs = all_sessions[rid]
                note += f"- [[{rs['label']}]] ({rs['date']})\n"
        note += "\n"

    # Navigation
    note += f"""---

> [!tip] Navigation
> - [[Sessions Index|Back to Index]]
> - [[{s['type']}|All {s['type']} Sessions]]
"""

    return note


def create_index_note(sessions: list[dict]) -> str:
    """Create main index note."""
    # Group by month
    by_month = defaultdict(list)
    for s in sessions:
        month = s['date'][:7] if s['date'] != "Unknown" else "Unknown"
        by_month[month].append(s)

    note = """---
aliases: ["Index", "Home"]
---

# Sessions Index

> [!summary] Overview
> Total sessions: {total}
> Period: {start} to {end}

## By Month

""".format(
        total=len(sessions),
        start=sessions[0]['date'] if sessions else "N/A",
        end=sessions[-1]['date'] if sessions else "N/A"
    )

    for month in sorted(by_month.keys(), reverse=True):
        note += f"### {month}\n\n"
        for s in sorted(by_month[month], key=lambda x: x['date'], reverse=True):
            note += f"- [[{s['label']}]] - {s['type']}\n"
        note += "\n"

    # By type
    note += "## By Type\n\n"
    by_type = defaultdict(list)
    for s in sessions:
        by_type[s['type']].append(s)

    for typ in sorted(by_type.keys()):
        note += f"- [[{typ}]] ({len(by_type[typ])} sessions)\n"

    # By topic
    note += "\n## By Topic\n\n"
    all_topics = set()
    for s in sessions:
        all_topics.update(s['topics'])

    for topic in sorted(all_topics):
        count = sum(1 for s in sessions if topic in s['topics'])
        note += f"- [[{topic}]] ({count})\n"

    return note


def create_topic_note(topic: str, sessions: list[dict]) -> str:
    """Create note for a topic."""
    relevant = [s for s in sessions if topic in s['topics']]

    note = f"""---
type: topic
---

# {topic}

> [!info] {len(relevant)} sessions related to this topic

## Sessions

"""
    for s in sorted(relevant, key=lambda x: x['date'], reverse=True):
        note += f"- [[{s['label']}]] ({s['date']}) - {s['type']}\n"

    note += "\n## Related Topics\n\n"

    # Find co-occurring topics
    co_topics = defaultdict(int)
    for s in relevant:
        for t in s['topics']:
            if t != topic:
                co_topics[t] += 1

    for t, count in sorted(co_topics.items(), key=lambda x: -x[1])[:10]:
        note += f"- [[{t}]] ({count} shared)\n"

    return note


def create_type_note(session_type: str, sessions: list[dict]) -> str:
    """Create note for session type."""
    relevant = [s for s in sessions if s['type'] == session_type]

    note = f"""---
type: session-type
---

# {session_type} Sessions

> [!info] {len(relevant)} sessions of this type

## All Sessions

"""
    for s in sorted(relevant, key=lambda x: x['date'], reverse=True):
        topics_str = ', '.join(s['topics'][:3])
        note += f"- [[{s['label']}]] ({s['date']}) - {topics_str}\n"

    return note


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

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "Sessions").mkdir(exist_ok=True)
    (OUTPUT_PATH / "Topics").mkdir(exist_ok=True)
    (OUTPUT_PATH / "Types").mkdir(exist_ok=True)

    # Find relations
    print("Finding relations...")
    relations = find_related_by_topics(sessions)
    all_sessions = {s['id']: s for s in sessions}

    # Create session notes
    print("Creating session notes...")
    for s in sessions:
        note = create_session_note(s, relations, all_sessions)
        filepath = OUTPUT_PATH / "Sessions" / f"{s['label']}.md"
        filepath.write_text(note, encoding='utf-8')

    # Create index
    print("Creating index...")
    index = create_index_note(sessions)
    (OUTPUT_PATH / "Sessions Index.md").write_text(index, encoding='utf-8')

    # Create topic notes
    print("Creating topic notes...")
    all_topics = set()
    for s in sessions:
        all_topics.update(s['topics'])

    for topic in all_topics:
        note = create_topic_note(topic, sessions)
        filepath = OUTPUT_PATH / "Topics" / f"{topic}.md"
        filepath.write_text(note, encoding='utf-8')

    # Create type notes
    print("Creating type notes...")
    types = set(s['type'] for s in sessions)
    for t in types:
        note = create_type_note(t, sessions)
        filepath = OUTPUT_PATH / "Types" / f"{t}.md"
        filepath.write_text(note, encoding='utf-8')

    print(f"\nExported to: {OUTPUT_PATH}")
    print(f"  - {len(sessions)} session notes")
    print(f"  - {len(all_topics)} topic notes")
    print(f"  - {len(types)} type notes")
    print(f"  - 1 index note")
    print("\nOpen this folder as an Obsidian vault!")


if __name__ == "__main__":
    main()
