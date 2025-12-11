# Code Style and Conventions

## Python Version
- **Required**: Python 3.11
- Always use `py -3.11` on Windows

## Naming Conventions
- **Classes**: PascalCase (e.g., `PipelineConfig`, `DETRPipelineOrchestrator`)
- **Functions/Methods**: snake_case (e.g., `run_pipeline`, `load_model`)
- **Variables**: snake_case (e.g., `video_path`, `output_dir`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_EPOCHS`, `DEFAULT_THRESHOLD`)
- **Private methods**: Leading underscore (e.g., `_apply_dino_information_analysis`)

## Type Hints
Type hints are used for class attributes and function parameters:
```python
def __init__(self):
    self.video_directory: str = ""
    self.frames_per_video: int = 50
    self.clustering_threshold: float = 0.7

@classmethod
def from_yaml(cls, config_path: str) -> 'PipelineConfig':
    ...
```

## Docstrings
Triple-quoted docstrings for classes and methods:
```python
class PipelineConfig:
    """Configuration for the entire pipeline"""
    
def from_yaml(cls, config_path: str) -> 'PipelineConfig':
    """Load configuration from YAML file"""
```

## File Organization
- Configuration classes at top of file
- Main classes/logic in middle
- Entry point (`main()`) at bottom
- `if __name__ == "__main__":` pattern for scripts

## Logging
- Use Python's `logging` module
- Logger per module: `logger = logging.getLogger(__name__)`
- Standard log levels: DEBUG, INFO, WARNING, ERROR

## Configuration
- YAML files for configuration (`.yaml`)
- JSON for data interchange (`.json`)
- Dataclass-style configuration classes with validation

## Error Handling
- Explicit validation methods (e.g., `config.validate()`)
- Collect errors before raising
- Clear error messages with context

## Path Handling
- Use `pathlib.Path` for path operations
- Check paths exist before use
- Use UTF-8 encoding for file I/O

## Code Structure Pattern
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module docstring"""

import standard_lib
import third_party
import local_modules

logger = logging.getLogger(__name__)

class MyClass:
    """Class docstring"""
    
    def __init__(self):
        ...
    
    def public_method(self):
        """Method docstring"""
        ...
    
    def _private_method(self):
        ...

def main():
    """Main entry point"""
    ...

if __name__ == "__main__":
    main()
```
