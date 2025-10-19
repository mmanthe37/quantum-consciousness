"""
Dynamic connectivity analyzer for consciousness assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class DynamicConnectivityAnalyzer:
    """
    Analyzer for dynamic functional connectivity patterns.
    """
    
    def __init__(self):
        pass
    
    def analyze_dynamic_patterns(self, data: Any) -> Dict[str, float]:
        """Analyze dynamic connectivity patterns."""
        results = {}
        
        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
            results['dynamic_connectivity_score'] = self._calculate_dynamic_connectivity_score(data)
            results['integration_measure'] = self._calculate_integration_measure(data)
            results['differentiation_measure'] = self._calculate_differentiation_measure(data)
        
        return results
    
    def _calculate_dynamic_connectivity_score(self, data: np.ndarray) -> float:
        """Calculate dynamic connectivity score."""
        if data.size == 0:
            return 0.0
        
        # Calculate time-varying correlation
        if len(data.shape) < 2:
            return 0.0
        
        window_size = min(50, data.shape[1] // 4)
        if window_size < 5:
            return 0.0
        
        connectivity_values = []
        
        for start in range(0, data.shape[1] - window_size, window_size // 2):
            end = start + window_size
            window_data = data[:, start:end]
            
            # Calculate mean connectivity in this window
            corr_matrix = np.corrcoef(window_data)
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            mean_connectivity = np.mean(np.abs(upper_triangle))
            connectivity_values.append(mean_connectivity)
        
        # Dynamic connectivity as variance in connectivity over time
        if len(connectivity_values) > 1:
            return float(np.var(connectivity_values))
        else:
            return 0.0
    
    def _calculate_integration_measure(self, data: np.ndarray) -> float:
        """Calculate integration measure."""
        if data.size == 0 or len(data.shape) < 2:
            return 0.0
        
        # Integration as mean correlation strength
        corr_matrix = np.corrcoef(data)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        integration = np.mean(np.abs(upper_triangle))
        
        return float(integration)
    
    def _calculate_differentiation_measure(self, data: np.ndarray) -> float:
        """Calculate differentiation measure."""
        if data.size == 0:
            return 0.0
        
        # Differentiation as variance across channels/features
        if len(data.shape) > 1:
            channel_variances = np.var(data, axis=1)
            differentiation = np.mean(channel_variances)
        else:
            differentiation = np.var(data)
        
        return float(differentiation)