#!/bin/bash

# =============================================================================
# List Sessions - Helper Tool
# =============================================================================
# Purpose: List all sessions by type with quick stats
# Usage: ./list_sessions.sh [--type TYPE] [--recent N]
# =============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
FILTER_TYPE=""
RECENT_COUNT=10

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSIONS_ROOT="$PROJECT_ROOT/.sessions"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            FILTER_TYPE="$2"
            shift 2
            ;;
        --recent)
            RECENT_COUNT="$2"
            shift 2
            ;;
        --help)
            cat << EOF
Usage: ./list_sessions.sh [OPTIONS]

List all sessions with statistics.

OPTIONS:
    --type TYPE     Filter by session type: ssh, benchmark, analysis, training
    --recent N      Show only N most recent sessions (default: 10, use 0 for all)
    --help          Show this help message

EXAMPLES:
    # List all sessions
    ./list_sessions.sh

    # List only SSH sessions
    ./list_sessions.sh --type ssh

    # List 5 most recent sessions
    ./list_sessions.sh --recent 5

    # List all benchmark sessions
    ./list_sessions.sh --type benchmark --recent 0
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${CYAN}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Session Manager - List Sessions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# Count sessions by type
count_ssh=$(find "$SESSIONS_ROOT/ssh" -maxdepth 1 -type d -name "sesja_*" 2>/dev/null | wc -l)
count_benchmark=$(find "$SESSIONS_ROOT/benchmark" -maxdepth 1 -type d -name "sesja_*" 2>/dev/null | wc -l)
count_analysis=$(find "$SESSIONS_ROOT/analysis" -maxdepth 1 -type d -name "sesja_*" 2>/dev/null | wc -l)
count_training=$(find "$SESSIONS_ROOT/training" -maxdepth 1 -type d -name "sesja_*" 2>/dev/null | wc -l)
count_total=$((count_ssh + count_benchmark + count_analysis + count_training))

# Print statistics
echo -e "${BLUE}📊 Session Statistics:${NC}"
echo "  Total Sessions: $count_total"
echo "  SSH: $count_ssh"
echo "  Benchmark: $count_benchmark"
echo "  Analysis: $count_analysis"
echo "  Training: $count_training"
echo ""

# List sessions
if [ -z "$FILTER_TYPE" ]; then
    # List all types
    TYPES=("ssh" "benchmark" "analysis" "training")
else
    # List only specified type
    TYPES=("$FILTER_TYPE")
fi

for TYPE in "${TYPES[@]}"; do
    TYPE_DIR="$SESSIONS_ROOT/$TYPE"

    if [ ! -d "$TYPE_DIR" ]; then
        continue
    fi

    # Find sessions (newest first)
    sessions=$(find "$TYPE_DIR" -maxdepth 1 -type d -name "sesja_*" -printf "%T@ %p\n" 2>/dev/null | sort -rn | cut -d' ' -f2-)

    if [ -z "$sessions" ]; then
        continue
    fi

    # Count sessions for this type
    session_count=$(echo "$sessions" | wc -l)

    echo -e "${GREEN}═══ ${TYPE^^} ($session_count sessions) ═══${NC}"
    echo ""

    # Limit to recent if requested
    if [ "$RECENT_COUNT" -gt 0 ]; then
        sessions=$(echo "$sessions" | head -n "$RECENT_COUNT")
    fi

    # List sessions
    while IFS= read -r session_path; do
        session_name=$(basename "$session_path")
        readme_path="$session_path/README.md"

        # Extract description if README exists
        description="(no description)"
        if [ -f "$readme_path" ]; then
            description=$(grep "^**Description:**" "$readme_path" | sed 's/^**Description:** //' || echo "(no description)")
        fi

        # Get last modified time
        mod_time=$(stat -c %y "$session_path" 2>/dev/null || stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$session_path" 2>/dev/null || echo "unknown")

        echo -e "${YELLOW}▸${NC} $session_name"
        echo "  Description: $description"
        echo "  Modified: $mod_time"
        echo ""
    done <<< "$sessions"
done

# Navigation hint
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}💡 Tip:${NC} Navigate to session: cd .sessions/[TYPE]/sesja_..."
echo ""
