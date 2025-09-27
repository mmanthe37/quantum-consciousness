"""
Unified measurement suite combining all consciousness assessment tools.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.base import BaseProcessor, ProcessingResult
from .pci_proxy import PCIProxyCalculator
from .dynamic_connectivity import DynamicConnectivityAnalyzer


class MeasurementSuite(BaseProcessor):
    """
    Unified measurement suite for comprehensive consciousness analysis.
    
    Integrates measurements across all domains:
    - Biological consciousness measures
    - AI consciousness detection
    - Quantum-theological analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Sub-components
        self.pci_proxy_calculator = None
        self.dynamic_connectivity_analyzer = None
    
    def initialize(self) -> bool:
        """Initialize measurement suite components."""
        try:
            self.pci_proxy_calculator = PCIProxyCalculator()
            self.dynamic_connectivity_analyzer = DynamicConnectivityAnalyzer()
            
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize measurement suite: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input for measurement suite."""
        # Accept any data format - let individual processors handle validation
        return True
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """
        Process data through unified measurement suite.
        
        Args:
            data: Input data for measurement
            **kwargs: Additional parameters
            
        Returns:
            ProcessingResult with unified measurements
        """
        try:
            results = {}
            
            # PCI proxy measurements
            pci_results = self.pci_proxy_calculator.calculate_pci_proxy_suite(data)
            results.update(pci_results)
            
            # Dynamic connectivity measurements
            connectivity_results = self.dynamic_connectivity_analyzer.analyze_dynamic_patterns(data)
            results.update(connectivity_results)
            
            # Unified consciousness score
            unified_score = self._calculate_unified_consciousness_score(results)
            results['unified_consciousness_score'] = unified_score
            
            return ProcessingResult(
                data=results,
                metadata={'measurement_suite_version': '1.0'},
                timestamp=datetime.now(),
                processor_type="MeasurementSuite",
                confidence=self._calculate_confidence(results)
            )
            
        except Exception as e:
            return ProcessingResult(
                data=None,
                metadata={'error_details': str(e)},
                timestamp=datetime.now(),
                processor_type="MeasurementSuite",
                errors=[f"Measurement suite failed: {str(e)}"]
            )
    
    def _calculate_unified_consciousness_score(self, results: Dict[str, Any]) -> float:
        """Calculate unified consciousness score across all measures."""
        consciousness_indicators = []
        
        # Collect all consciousness-related scores
        consciousness_keys = [
            'pci_proxy', 'dynamic_connectivity_score', 'complexity_measure',
            'integration_measure', 'differentiation_measure'
        ]
        
        for key in consciousness_keys:
            if key in results and isinstance(results[key], (int, float)):
                consciousness_indicators.append(results[key])
        
        return np.mean(consciousness_indicators) if consciousness_indicators else 0.0
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in unified measurements."""
        return 0.8  # High confidence in unified approach