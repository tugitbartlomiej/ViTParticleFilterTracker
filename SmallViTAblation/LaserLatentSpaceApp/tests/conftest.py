"""Wspólne ustawienia testów aplikacji: katalog aplikacji w sys.path (pakiet laser_app), bez pliku ustawień, konteksty OpenGL
współdzielone przed pierwszą QApplication (jak w run.py).

Bez współdzielonych kontekstów widok OpenGL utworzony w późniejszym teście dostaje osobny kontekst, a programy shaderów
pyqtgraph z wcześniejszego widoku są w nim nieważne (GL_INVALID_VALUE przy glUseProgram)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # LaserLatentSpaceApp: pakiet laser_app

os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
os.environ.setdefault('EDYTOR_BEZ_AUTOSTARTU', '1')

import laser_app  # noqa: E402,F401  (wyłącza sprawdzanie błędów PyOpenGL)
from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
