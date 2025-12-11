<#
.SYNOPSIS
    Session Manager - PowerShell Script

.DESCRIPTION
    Create timestamped session folders with templates (Windows-optimized)

.PARAMETER Type
    Session type: ssh, benchmark, analysis, training

.PARAMETER Description
    Session description

.PARAMETER Files
    Comma-separated paths to files to copy

.PARAMETER OpenInEditor
    Open session folder in default editor after creation

.EXAMPLE
    .\save_session.ps1 -Type ssh -Description "DETR training Job 1226363"

.EXAMPLE
    .\save_session.ps1 -Type benchmark -Description "YOLO vs DETR" -Files "..\results\*.json" -OpenInEditor

.EXAMPLE
    .\save_session.ps1 -Type analysis -Description "Query 81 analysis"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("ssh", "benchmark", "analysis", "training")]
    [string]$Type,

    [Parameter(Mandatory=$true)]
    [string]$Description,

    [Parameter(Mandatory=$false)]
    [string]$Files,

    [Parameter(Mandatory=$false)]
    [switch]$OpenInEditor
)

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Header {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Session Manager - Save Session (PowerShell)" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "√ " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Host "× " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "i " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Get-UTCTimestamp {
    return (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd_HH-mm-ss")
}

function New-SessionFolder {
    param(
        [string]$SessionsRoot,
        [string]$SessionType
    )

    $timestamp = Get-UTCTimestamp
    $sessionName = "sesja_${timestamp}_${SessionType}"
    $sessionPath = Join-Path $SessionsRoot $SessionType $sessionName

    # Handle conflicts
    if (Test-Path $sessionPath) {
        Write-Warning "Session folder already exists: $sessionPath"
        Write-Info "Adding timestamp suffix to avoid conflict"
        $sessionName = "${sessionName}_$([DateTimeOffset]::Now.ToUnixTimeSeconds())"
        $sessionPath = Join-Path $SessionsRoot $SessionType $sessionName
    }

    New-Item -ItemType Directory -Path $sessionPath -Force | Out-Null
    return $sessionPath
}

function New-ReadmeFile {
    param(
        [string]$SessionPath,
        [string]$SessionType,
        [string]$Description
    )

    $sessionName = Split-Path $SessionPath -Leaf
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss")

    $readme = @"
# Session: $sessionName

**Session Type:** $SessionType
**Created:** $timestamp UTC
**Description:** $Description

## Quick Reference
- **Folder:** ``$sessionName``
- **Type:** $SessionType
- **Status:** In Progress

## Files in This Session
- ``SESSION_SUMMARY.md`` - Complete session details
- ``README.md`` - This file

## How to Review
1. Open ``SESSION_SUMMARY.md`` for full context
2. Check session-specific files for data and results

## Related Sessions
- **Previous:** (to be filled)
- **Next:** (to be filled)

## Commands

### Navigate to session
``````powershell
cd "$SessionPath"
``````

### List all files
``````powershell
Get-ChildItem "$SessionPath"
``````

---

**Session Created:** $timestamp UTC
**Last Updated:** $timestamp UTC
"@

    $readmePath = Join-Path $SessionPath "README.md"
    [System.IO.File]::WriteAllText($readmePath, $readme, [System.Text.Encoding]::UTF8)
    Write-Success "Created README.md"
}

function New-SessionSummary {
    param(
        [string]$SessionPath,
        [string]$SessionType,
        [string]$Description
    )

    $sessionName = Split-Path $SessionPath -Leaf
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss")
    $datePart = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
    $timePart = (Get-Date).ToUniversalTime().ToString("HH:mm:ss")

    $summary = @"
# Session Summary: $sessionName

## Metadata
- **Type:** $SessionType
- **Date:** $datePart
- **Time:** $timePart UTC
- **Duration:** (to be filled)
- **Status:** In Progress

## Objective
$Description

## Context
(Provide background information, previous work, why this session was needed)

## Actions Taken

### Step 1: (Fill in action title)
**Description:** (Describe what was done)

**Command:**
``````bash
# Commands used
``````

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
- ``file1.txt`` - Description
- ``file2.json`` - Description

## Commands Used
``````bash
# List commands used in this session
``````

## Related Work
- Previous session: (link)
- Related analysis: (link)

---

**Session Created:** $timestamp UTC
**Last Updated:** $timestamp UTC
"@

    $summaryPath = Join-Path $SessionPath "SESSION_SUMMARY.md"
    [System.IO.File]::WriteAllText($summaryPath, $summary, [System.Text.Encoding]::UTF8)
    Write-Success "Created SESSION_SUMMARY.md"
}

function Copy-SessionFiles {
    param(
        [string]$SessionPath,
        [string]$FilesPattern
    )

    if ([string]::IsNullOrEmpty($FilesPattern)) {
        Write-Info "No files specified to copy"
        return
    }

    Write-Info "Copying specified files..."

    $fileList = $FilesPattern -split ','
    foreach ($filePattern in $fileList) {
        $filePattern = $filePattern.Trim()

        # Resolve paths
        $resolvedFiles = Resolve-Path $filePattern -ErrorAction SilentlyContinue

        if ($resolvedFiles) {
            foreach ($file in $resolvedFiles) {
                if (Test-Path $file -PathType Container) {
                    # Copy directory
                    Copy-Item -Path $file -Destination $SessionPath -Recurse -Force
                    Write-Success "Copied directory: $file"
                } else {
                    # Copy file
                    Copy-Item -Path $file -Destination $SessionPath -Force
                    Write-Success "Copied file: $file"
                }
            }
        } else {
            Write-Warning "File not found: $filePattern"
        }
    }
}

# ============================================================================
# Main Script
# ============================================================================

Write-Header

# Get project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$sessionsRoot = Join-Path $projectRoot ".sessions"

Write-Info "Creating new $Type session..."
Write-Info "Description: $Description"
Write-Host ""

# Create session folder
$sessionPath = New-SessionFolder -SessionsRoot $sessionsRoot -SessionType $Type
Write-Success "Created session folder: $sessionPath"

# Create README and summary
New-ReadmeFile -SessionPath $sessionPath -SessionType $Type -Description $Description
New-SessionSummary -SessionPath $sessionPath -SessionType $Type -Description $Description

# Copy files if specified
Copy-SessionFiles -SessionPath $sessionPath -FilesPattern $Files

# Open in editor if requested
if ($OpenInEditor) {
    Write-Info "Opening session in default editor..."
    Start-Process $sessionPath
}

# Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Green
Write-Success "Session created successfully!"
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Info "Session Path: $sessionPath"
Write-Info "Session Name: $(Split-Path $sessionPath -Leaf)"
Write-Host ""
Write-Info "Next steps:"
Write-Host "  1. cd `"$sessionPath`""
Write-Host "  2. Edit SESSION_SUMMARY.md to document your work"
Write-Host "  3. Add files to the session folder as needed"
Write-Host ""
