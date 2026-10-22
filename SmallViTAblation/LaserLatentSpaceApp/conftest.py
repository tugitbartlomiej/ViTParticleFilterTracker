"""Wspólne ustawienia testów aplikacji: jak w start.py, konteksty OpenGL współdzielone (przed pierwszą QApplication).

Bez tego widok OpenGL utworzony w późniejszym teście dostaje osobny kontekst, a programy shaderów pyqtgraph
z wcześniejszego widoku są w nim nieważne (GL_INVALID_VALUE przy glUseProgram)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # moduły aplikacji dla testów w testy/

os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
os.environ.setdefault('EDYTOR_BEZ_AUTOSTARTU', '1')

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
