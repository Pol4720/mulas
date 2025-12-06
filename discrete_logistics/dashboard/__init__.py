"""
Dashboard Module - Interactive Web Application
==============================================

This module provides a complete interactive web dashboard
for the Balanced Multi-Bin Packing problem using Streamlit.

Components:
-----------
- app: Main Streamlit application
- pages: Individual dashboard pages
- components: Reusable UI components
"""

from .components import (
    ProblemConfigurator,
    AlgorithmSelector,
    ResultsDisplay,
    VisualizationPanel
)

__all__ = [
    'ProblemConfigurator',
    'AlgorithmSelector', 
    'ResultsDisplay',
    'VisualizationPanel'
]
