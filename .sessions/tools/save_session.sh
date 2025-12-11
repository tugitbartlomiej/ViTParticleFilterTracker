#!/bin/bash

# =============================================================================
# Session Manager - Bash Script
# =============================================================================
# Purpose: Create timestamped session folders with templates
# Usage: ./save_session.sh --type [ssh|benchmark|analysis|training] --desc "description"
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
SESSION_TYPE=""
DESCRIPTION=""
FILES_TO_COPY=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSIONS_ROOT="$PROJECT_ROOT/.sessions"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${CYAN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Session Manager - Save Session"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

show_help() {
    cat << EOF
Usage: ./save_session.sh [OPTIONS]

Create a new timestamped session folder with templates.

OPTIONS:
    --type TYPE         Session type: ssh, benchmark, analysis, training (REQUIRED)
    --desc DESCRIPTION  Session description (REQUIRED)
    --files FILES       Comma-separated paths to files to copy (OPTIONAL)
    --help              Show this help message

EXAMPLES:
    # Create SSH session
    ./save_session.sh --type ssh --desc "DETR training Job 1226363"

    # Create benchmark session with files
    ./save_session.sh --type benchmark \\
        --desc "YOLO vs DETR epoch 140" \\
        --files "../YOLO_DETR_Benchmarks/results/*.json,../configs/*.yaml"

    # Create analysis session
    ./save_session.sh --type analysis --desc "Query 81 specialization"

SESSION TYPES:
    ssh         - SSH cluster work (Eden, etc.)
    benchmark   - Model comparison and performance testing
    analysis    - Data exploration and investigation
    training    - Local model training sessions

EOF
}

validate_session_type() {
    case "$SESSION_TYPE" in
        ssh|benchmark|analysis|training)
            return 0
            ;;
        *)
            print_error "Invalid session type: $SESSION_TYPE"
            echo "Valid types: ssh, benchmark, analysis, training"
            return 1
            ;;
    esac
}

get_utc_timestamp() {
    date -u +"%Y-%m-%d_%H-%M-%S"
}

create_session_folder() {
    local timestamp=$(get_utc_timestamp)
    local session_name="sesja_${timestamp}_${SESSION_TYPE}"
    local session_path="$SESSIONS_ROOT/$SESSION_TYPE/$session_name"

    if [ -d "$session_path" ]; then
        print_warning "Session folder already exists: $session_path"
        print_info "Adding timestamp suffix to avoid conflict"
        session_name="${session_name}_$(date +%s)"
        session_path="$SESSIONS_ROOT/$SESSION_TYPE/$session_name"
    fi

    mkdir -p "$session_path"
    echo "$session_path"
}

create_readme() {
    local session_path="$1"
    local session_name=$(basename "$session_path")
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S")

    cat > "$session_path/README.md" << EOF
# Session: $session_name

**Session Type:** $SESSION_TYPE
**Created:** $timestamp UTC
**Description:** $DESCRIPTION

## Quick Reference
- **Folder:** \`$session_name\`
- **Type:** $SESSION_TYPE
- **Status:** In Progress

## Files in This Session
- \`SESSION_SUMMARY.md\` - Complete session details
- \`README.md\` - This file

## How to Review
1. Open \`SESSION_SUMMARY.md\` for full context
2. Check session-specific files for data and results

## Related Sessions
- **Previous:** (to be filled)
- **Next:** (to be filled)

## Commands

### Navigate to session
\`\`\`bash
cd "$session_path"
\`\`\`

### List all files
\`\`\`bash
ls -la "$session_path"
\`\`\`

---

**Session Created:** $timestamp UTC
**Last Updated:** $timestamp UTC
EOF

    print_success "Created README.md"
}

create_session_summary() {
    local session_path="$1"
    local session_name=$(basename "$session_path")
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S")
    local date_part=$(date -u +"%Y-%m-%d")
    local time_part=$(date -u +"%H:%M:%S")

    cat > "$session_path/SESSION_SUMMARY.md" << EOF
# Session Summary: $session_name

## Metadata
- **Type:** $SESSION_TYPE
- **Date:** $date_part
- **Time:** $time_part UTC
- **Duration:** (to be filled)
- **Status:** In Progress

## Objective
$DESCRIPTION

## Context
(Provide background information, previous work, why this session was needed)

## Actions Taken

### Step 1: (Fill in action title)
**Description:** (Describe what was done)

**Command:**
\`\`\`bash
# Commands used
\`\`\`

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
- \`file1.txt\` - Description
- \`file2.json\` - Description

## Commands Used
\`\`\`bash
# List commands used in this session
\`\`\`

## Related Work
- Previous session: (link)
- Related analysis: (link)

---

**Session Created:** $timestamp UTC
**Last Updated:** $timestamp UTC
EOF

    print_success "Created SESSION_SUMMARY.md"
}

copy_files() {
    local session_path="$1"

    if [ -z "$FILES_TO_COPY" ]; then
        print_info "No files specified to copy"
        return 0
    fi

    print_info "Copying specified files..."

    # Split comma-separated file list
    IFS=',' read -ra FILE_ARRAY <<< "$FILES_TO_COPY"

    for file_pattern in "${FILE_ARRAY[@]}"; do
        # Expand glob patterns
        for file in $file_pattern; do
            if [ -e "$file" ]; then
                if [ -d "$file" ]; then
                    # Copy directory
                    cp -r "$file" "$session_path/"
                    print_success "Copied directory: $file"
                else
                    # Copy file
                    cp "$file" "$session_path/"
                    print_success "Copied file: $file"
                fi
            else
                print_warning "File not found: $file"
            fi
        done
    done
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    print_header

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                SESSION_TYPE="$2"
                shift 2
                ;;
            --desc)
                DESCRIPTION="$2"
                shift 2
                ;;
            --files)
                FILES_TO_COPY="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [ -z "$SESSION_TYPE" ] || [ -z "$DESCRIPTION" ]; then
        print_error "Missing required arguments"
        echo ""
        show_help
        exit 1
    fi

    # Validate session type
    if ! validate_session_type; then
        exit 1
    fi

    print_info "Creating new $SESSION_TYPE session..."
    print_info "Description: $DESCRIPTION"
    echo ""

    # Create session folder
    SESSION_PATH=$(create_session_folder)
    print_success "Created session folder: $SESSION_PATH"

    # Create README
    create_readme "$SESSION_PATH"

    # Create SESSION_SUMMARY
    create_session_summary "$SESSION_PATH"

    # Copy files if specified
    copy_files "$SESSION_PATH"

    # Summary
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    print_success "Session created successfully!"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    print_info "Session Path: $SESSION_PATH"
    print_info "Session Name: $(basename "$SESSION_PATH")"
    echo ""
    print_info "Next steps:"
    echo "  1. cd \"$SESSION_PATH\""
    echo "  2. Edit SESSION_SUMMARY.md to document your work"
    echo "  3. Add files to the session folder as needed"
    echo ""
}

main "$@"
