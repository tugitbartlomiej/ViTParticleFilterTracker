# Task Completion Checklist

## Before Starting
- [ ] Read relevant existing code
- [ ] Understand the task requirements
- [ ] Check if similar patterns exist in codebase

## During Development
- [ ] Follow Python 3.11 compatibility
- [ ] Use type hints for parameters
- [ ] Add docstrings for public methods
- [ ] Follow snake_case naming convention
- [ ] Handle paths with pathlib.Path
- [ ] Use UTF-8 encoding for file I/O

## Code Quality
- [ ] No hardcoded paths (use config)
- [ ] Proper error handling with context
- [ ] Logging at appropriate levels
- [ ] No unused imports or variables

## Testing
- [ ] Test script runs without errors
- [ ] Test with sample data if available
- [ ] Check GPU usage if applicable

## Before Commit
- [ ] Run the modified scripts
- [ ] Check for syntax errors
- [ ] Verify paths are correct
- [ ] Test on Windows (primary platform)

## No Formal Linting/Formatting Tools Configured
The project does not have:
- `.flake8` config
- `pytest.ini`
- `.pre-commit-config.yaml`
- Black/isort configuration

Manual code review is the primary quality check.

## Git Workflow
```bash
# Check status
git status

# Add and commit
git add <files>
git commit -m "descriptive message"

# Push to feature branch
git push origin feature/branch-name
```

## Common Issues to Avoid
1. Using wrong Python version (must be 3.11)
2. Hardcoded Windows paths
3. Missing CUDA checks before GPU operations
4. Not handling missing files/directories
5. Forgetting UTF-8 encoding
