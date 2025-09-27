"""
Quantum Consciousness Research Framework

A comprehensive multi-scale framework for studying consciousness across
biological, digital, and quantum domains, integrating:
- Integrated Information Theory (IIT)
- AI consciousness detection algorithms
- Biological measurement protocols (EEG, fMRI)
- Quantum-theological analysis tools
"""

__version__ = "0.1.0"
__author__ = "OmniSphere"
__email__ = "research@omnisphere.org"

from .core import QuantumConsciousnessFramework
from .iit import IITProcessor
from .ai_consciousness import AIConsciousnessDetector
from .biological import BiologicalProcessor
from .quantum_theological import QuantumTheologicalAnalyzer
from .measurement import MeasurementSuite
from .analytical import AnalyticalEngine

__all__ = [
    "QuantumConsciousnessFramework",
    "IITProcessor", 
    "AIConsciousnessDetector",
    "BiologicalProcessor",
    "QuantumTheologicalAnalyzer",
    "MeasurementSuite",
    "AnalyticalEngine",
]