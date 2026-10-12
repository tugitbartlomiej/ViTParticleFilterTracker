"""Uruchamia używaną wcześniej bibliotekę ManimGL, bez zmieniania jej instalacji."""
import sys
import runpy
from pathlib import Path

LIB=Path(r'F:\Studia\MCHTR\Grant\Projekt\Research\Prezentacja\Inne\manim\Animation_codes\.venv-manimgl\Lib\site-packages')
sys.path.append(str(LIB))
runpy.run_module('manimlib',run_name='__main__')
