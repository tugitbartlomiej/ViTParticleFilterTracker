"""Generate session data with short labels for vertical tree."""

import re
import json
from pathlib import Path
from datetime import datetime

SESSIONS_PATH = Path(__file__).parent.parent


def generate_short_label(name: str, content: str) -> str:
    """Generate 1-2 word label from session."""

    # Try to extract key topic
    patterns = [
        (r'YOLO.*Resume|Resume.*YOLO', 'YOLO Resume'),
        (r'YOLO.*Fix|Fix.*YOLO', 'YOLO Fix'),
        (r'YOLO.*LR', 'YOLO LR'),
        (r'YOLO.*Training', 'YOLO Train'),
        (r'DETR.*EL2N|EL2N.*DETR', 'DETR EL2N'),
        (r'DETR.*Query|Query.*81', 'Query 81'),
        (r'IEEE.*Article|Article.*IEEE', 'IEEE Article'),
        (r'IEEE.*Pipeline', 'IEEE Pipeline'),
        (r'IEEE.*Cleanup', 'IEEE Cleanup'),
        (r'IEEE.*Verification', 'IEEE Verify'),
        (r'Dataset.*Selection', 'Dataset Select'),
        (r'Scientific.*Justification', 'Justification'),
        (r'PyTorch.*GPU|GPU.*Fix', 'GPU Fix'),
        (r'Benchmark.*Fix', 'Bench Fix'),
        (r'Advanced.*Dataset', 'Adv Dataset'),
        (r'SSH.*Eden|Eden.*SSH', 'SSH Eden'),
    ]

    combined = f"{name} {content[:500]}"
    for pattern, label in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return label

    # Fallback: extract key words from name
    name_clean = re.sub(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_?', '', name)
    name_clean = name_clean.replace('_', ' ').strip()

    if name_clean:
        words = name_clean.split()[:2]
        return ' '.join(words)

    # Last resort: use type detection
    if 'training' in content.lower():
        return 'Training'
    if 'benchmark' in content.lower():
        return 'Benchmark'
    if 'analysis' in content.lower():
        return 'Analysis'

    return 'Session'


def generate_semantic_summary(content: str, name: str) -> str:
    """Generate short semantic summary of session."""

    # Try to extract key results/findings
    results = []

    # Look for Key Findings section
    findings_match = re.search(r'### Key Findings\n(.+?)(?=\n###|\n##|\Z)', content, re.DOTALL)
    if findings_match:
        findings = findings_match.group(1)
        items = re.findall(r'[-*]\s*\*\*(.+?)\*\*', findings)
        if items:
            results.extend(items[:2])

    # Look for Results section
    if not results:
        results_match = re.search(r'## Results\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if results_match:
            items = re.findall(r'[-*]\s+(.+?)(?:\n|$)', results_match.group(1))
            results.extend([i[:60] for i in items[:2] if len(i) > 10])

    # Look for problems solved
    if not results:
        problems_match = re.search(r'Problem.*?:\s*(.+?)(?:\n|$)', content)
        if problems_match:
            results.append(f"Fixed: {problems_match.group(1)[:50]}")

    # Look for Actions Taken
    if not results:
        actions_match = re.search(r'## Actions Taken\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if actions_match:
            items = re.findall(r'\d+\.\s+\*\*(.+?)\*\*', actions_match.group(1))
            if items:
                results.extend(items[:2])

    # Extract metrics if present
    metrics = re.findall(r'(\d+(?:\.\d+)?%|\d+\.\d+\s*(?:mAP|loss|accuracy))', content[:1500])
    if metrics:
        results.append(f"Metrics: {', '.join(metrics[:3])}")

    # Build summary
    if results:
        summary = '. '.join(results[:2])
        if len(summary) > 120:
            summary = summary[:117] + '...'
        return summary

    # Fallback to objective
    obj_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if obj_match:
        obj = obj_match.group(1).strip()
        obj = re.sub(r'\s+', ' ', obj)
        return obj[:120] + '...' if len(obj) > 120 else obj

    return name[:80]


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

    # Name from folder
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder)
    name = desc_match.group(1).replace('_', ' ') if desc_match else folder

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
        objective = obj_match.group(1).strip()[:150]

    # Short label
    label = generate_short_label(name, content)

    # Semantic summary for hover
    semantic = generate_semantic_summary(content, name)

    # Extract full description for expanded view
    # Actions taken
    actions = []
    actions_match = re.search(r'## Actions Taken\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if actions_match:
        items = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\n##|\Z)', actions_match.group(1), re.DOTALL)
        actions = [re.sub(r'\s+', ' ', a.strip())[:100] for a in items[:5]]

    # Key findings
    findings = []
    findings_match = re.search(r'### Key Findings\n(.+?)(?=\n###|\n##|\Z)', content, re.DOTALL)
    if findings_match:
        items = re.findall(r'[-*]\s+(.+?)(?:\n|$)', findings_match.group(1))
        findings = [f[:80] for f in items[:4]]

    # Files modified
    files = []
    files_match = re.search(r'## Files\s+(?:Generated|Modified).*?\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if files_match:
        items = re.findall(r'`([^`]+\.(?:py|md|tex|yaml|json))`', files_match.group(1))
        files = items[:5]

    # Next steps
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
        "semantic": semantic,
        "actions": actions,
        "findings": findings,
        "files": files,
        "next_steps": next_steps
    }


def main():
    sessions = []
    for d in SESSIONS_PATH.iterdir():
        if d.is_dir() and d.name.startswith('Session_'):
            s = parse_session(d)
            if s:
                sessions.append(s)

    # Sort by date
    sessions.sort(key=lambda x: x["date"])

    # Read template
    template_path = SESSIONS_PATH / "analysis" / "session_tree_vertical.html"
    html = template_path.read_text(encoding='utf-8')

    # Replace placeholder
    json_data = json.dumps(sessions, indent=2, ensure_ascii=False)
    html = html.replace('SESSION_DATA_PLACEHOLDER', json_data)

    # Save
    output_path = SESSIONS_PATH / "analysis" / "session_tree.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"Generated: {output_path}")
    print(f"Sessions: {len(sessions)}")

    # Show labels
    print("\nLabels:")
    for s in sessions[-10:]:
        print(f"  {s['date'][:10]} | {s['label']}")


if __name__ == "__main__":
    main()
