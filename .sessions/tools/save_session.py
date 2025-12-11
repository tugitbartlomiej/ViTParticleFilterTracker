#!/usr/bin/env python3
"""
Session Manager - Python Script
================================
Purpose: Create timestamped session folders with templates
Usage: py -3.11 save_session.py --type [ssh|benchmark|analysis|training] --desc "description"
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    RESET = '\033[0m'

def print_header():
    """Print script header"""
    print(f"{Colors.CYAN}")
    print("━" * 50)
    print("  Session Manager - Save Session (Python)")
    print("━" * 50)
    print(f"{Colors.RESET}")

def print_success(msg: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")

def print_error(msg: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")

def print_info(msg: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")

def print_warning(msg: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")

def get_utc_timestamp() -> str:
    """Get current UTC timestamp in format YYYY-MM-DD_HH-MM-SS"""
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

def validate_session_type(session_type: str) -> bool:
    """Validate session type"""
    valid_types = ["ssh", "benchmark", "analysis", "training"]
    return session_type in valid_types

def create_session_folder(sessions_root: Path, session_type: str) -> Path:
    """Create timestamped session folder"""
    timestamp = get_utc_timestamp()
    session_name = f"sesja_{timestamp}_{session_type}"
    session_path = sessions_root / session_type / session_name

    # Handle conflicts
    if session_path.exists():
        print_warning(f"Session folder already exists: {session_path}")
        print_info("Adding timestamp suffix to avoid conflict")
        session_name = f"{session_name}_{int(datetime.now().timestamp())}"
        session_path = sessions_root / session_type / session_name

    session_path.mkdir(parents=True, exist_ok=True)
    return session_path

def create_readme(session_path: Path, session_type: str, description: str):
    """Create README.md for session"""
    session_name = session_path.name
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    readme_content = f"""# Session: {session_name}

**Session Type:** {session_type}
**Created:** {timestamp} UTC
**Description:** {description}

## Quick Reference
- **Folder:** `{session_name}`
- **Type:** {session_type}
- **Status:** In Progress

## Files in This Session
- `SESSION_SUMMARY.md` - Complete session details
- `README.md` - This file

## How to Review
1. Open `SESSION_SUMMARY.md` for full context
2. Check session-specific files for data and results

## Related Sessions
- **Previous:** (to be filled)
- **Next:** (to be filled)

## Commands

### Navigate to session
```bash
cd "{session_path}"
```

### List all files
```bash
ls -la "{session_path}"
```

---

**Session Created:** {timestamp} UTC
**Last Updated:** {timestamp} UTC
"""

    (session_path / "README.md").write_text(readme_content, encoding='utf-8')
    print_success("Created README.md")

def create_session_summary(session_path: Path, session_type: str, description: str):
    """Create SESSION_SUMMARY.md for session"""
    session_name = session_path.name
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    date_part = datetime.utcnow().strftime("%Y-%m-%d")
    time_part = datetime.utcnow().strftime("%H:%M:%S")

    summary_content = f"""# Session Summary: {session_name}

## Metadata
- **Type:** {session_type}
- **Date:** {date_part}
- **Time:** {time_part} UTC
- **Duration:** (to be filled)
- **Status:** In Progress

## Objective
{description}

## Context
(Provide background information, previous work, why this session was needed)

## Actions Taken

### Step 1: (Fill in action title)
**Description:** (Describe what was done)

**Command:**
```bash
# Commands used
```

**Result:**
(Describe the result)

## Key Findings
- Finding 1
- Finding 2
- Finding 3

## Issues Encountered

### Issue 1: (Issue title)
**Description:** (Describe the issue)
**Resolution:** (How it was resolved)

## Conclusions
(Summary of what was learned or achieved)

## Next Steps
- [ ] Next step 1
- [ ] Next step 2
- [ ] Next step 3

## Files Generated
- `file1.txt` - Description
- `file2.json` - Description

## Commands Used
```bash
# List commands used in this session
```

## Related Work
- Previous session: (link)
- Related analysis: (link)

---

**Session Created:** {timestamp} UTC
**Last Updated:** {timestamp} UTC
"""

    (session_path / "SESSION_SUMMARY.md").write_text(summary_content, encoding='utf-8')
    print_success("Created SESSION_SUMMARY.md")

def copy_files(session_path: Path, files: Optional[List[str]]):
    """Copy specified files to session folder"""
    if not files:
        print_info("No files specified to copy")
        return

    print_info("Copying specified files...")

    for file_pattern in files:
        # Handle glob patterns
        for file_path_str in file_pattern.split(','):
            file_path = Path(file_path_str.strip())

            if file_path.is_dir():
                # Copy directory
                dest = session_path / file_path.name
                shutil.copytree(file_path, dest, dirs_exist_ok=True)
                print_success(f"Copied directory: {file_path}")
            elif file_path.is_file():
                # Copy file
                shutil.copy2(file_path, session_path)
                print_success(f"Copied file: {file_path}")
            else:
                print_warning(f"File not found: {file_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create a new timestamped session folder with templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Create SSH session
    py -3.11 save_session.py --type ssh --desc "DETR training Job 1226363"

    # Create benchmark session with files
    py -3.11 save_session.py --type benchmark \\
        --desc "YOLO vs DETR epoch 140" \\
        --files "../YOLO_DETR_Benchmarks/results/*.json"

    # Create analysis session
    py -3.11 save_session.py --type analysis --desc "Query 81 specialization"

SESSION TYPES:
    ssh         - SSH cluster work (Eden, etc.)
    benchmark   - Model comparison and performance testing
    analysis    - Data exploration and investigation
    training    - Local model training sessions
        """
    )

    parser.add_argument("--type", required=True,
                        choices=["ssh", "benchmark", "analysis", "training"],
                        help="Session type")
    parser.add_argument("--desc", required=True,
                        help="Session description")
    parser.add_argument("--files", nargs='+',
                        help="Files or directories to copy (space-separated)")
    parser.add_argument("--open-vscode", action='store_true',
                        help="Open session in VSCode after creation")

    args = parser.parse_args()

    print_header()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    sessions_root = project_root / ".sessions"

    # Validate session type
    if not validate_session_type(args.type):
        print_error(f"Invalid session type: {args.type}")
        sys.exit(1)

    print_info(f"Creating new {args.type} session...")
    print_info(f"Description: {args.desc}")
    print()

    # Create session
    session_path = create_session_folder(sessions_root, args.type)
    print_success(f"Created session folder: {session_path}")

    # Create README and summary
    create_readme(session_path, args.type, args.desc)
    create_session_summary(session_path, args.type, args.desc)

    # Copy files if specified
    copy_files(session_path, args.files)

    # Open in VSCode if requested
    if args.open_vscode:
        print_info("Opening session in VSCode...")
        os.system(f'code "{session_path}"')

    # Summary
    print()
    print(f"{Colors.GREEN}{'━' * 50}{Colors.RESET}")
    print_success("Session created successfully!")
    print(f"{Colors.GREEN}{'━' * 50}{Colors.RESET}")
    print()
    print_info(f"Session Path: {session_path}")
    print_info(f"Session Name: {session_path.name}")
    print()
    print_info("Next steps:")
    print(f'  1. cd "{session_path}"')
    print("  2. Edit SESSION_SUMMARY.md to document your work")
    print("  3. Add files to the session folder as needed")
    print()

if __name__ == "__main__":
    main()
