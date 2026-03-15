"""Setup script for CataractAI Workbench."""

from setuptools import setup, find_packages

setup(
    name="CataractAI_Workbench",
    version="0.1.0",
    description="CataractAI Workbench - PyQt6 desktop application for cataract surgery ML pipeline",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "PyQt6>=6.5.0",
        "pyqtgraph>=0.13.0",
        "matplotlib>=3.7.0",
        "paramiko>=3.3.0",
        "scp>=0.14.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "cataract-workbench=CataractAI_Workbench.app.main:main",
        ],
    },
)
