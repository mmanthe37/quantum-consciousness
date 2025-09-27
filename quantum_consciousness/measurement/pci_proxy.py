"""
PCI proxy calculator for consciousness assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class PCIProxyCalculator:
    """
    Calculator for PCI proxy measures from various data types.
    """
    
    def __init__(self):
        pass
    
    def calculate_pci_proxy_suite(self, data: Any) -> Dict[str, float]:
        """Calculate comprehensive PCI proxy measures."""
        results = {}
        
        if isinstance(data, np.ndarray):
            results['pci_proxy'] = self._calculate_basic_pci_proxy(data)
            results['complexity_measure'] = self._calculate_complexity_measure(data)
        elif isinstance(data, dict):
            # Handle different data types
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    prefix = key[:3]  # Use first 3 chars as prefix
                    results[f'{prefix}_pci_proxy'] = self._calculate_basic_pci_proxy(value)
        
        return results
    
    def _calculate_basic_pci_proxy(self, data: np.ndarray) -> float:
        """Calculate basic PCI proxy from data."""
        if data.size == 0:
            return 0.0
        
        # Use Lempel-Ziv complexity as PCI proxy
        if len(data.shape) > 1:
            # Multi-channel: average across channels
            complexities = []
            for ch in range(min(data.shape[0], 10)):
                channel_data = data[ch]
                complexity = self._lempel_ziv_complexity(channel_data)
                complexities.append(complexity)
            return np.mean(complexities)
        else:
            return self._lempel_ziv_complexity(data)
    
    def _calculate_complexity_measure(self, data: np.ndarray) -> float:
        """Calculate general complexity measure."""
        if data.size == 0:
            return 0.0
        
        # Shannon entropy as complexity measure
        flat_data = data.flatten()
        hist, _ = np.histogram(flat_data, bins=20, density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        complexity = -np.sum(hist * np.log2(hist))
        
        return float(complexity)
    
    def _lempel_ziv_complexity(self, data: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity."""
        if len(data) == 0:
            return 0.0
        
        # Binarize data
        threshold = np.median(data)
        binary_data = (data > threshold).astype(int)
        
        # Calculate LZ complexity
        n = len(binary_data)
        complexity = 0
        i = 0
        
        while i < n:
            k = 1
            while i + k <= n:
                substring = binary_data[i:i+k]
                
                # Check if substring exists before
                found = False
                for j in range(i):
                    if j + k <= len(binary_data):
                        if np.array_equal(substring, binary_data[j:j+k]):
                            found = True
                            break
                
                if not found:
                    complexity += 1
                    break
                k += 1
                
                if k > n - i:
                    complexity += 1
                    break
            
            i += max(1, k)
        
        # Normalize
        return complexity / (n / np.log2(n)) if n > 1 else 0