"""
Main IIT processor for integrated information analysis.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..core.base import BaseProcessor, ProcessingResult, DataValidator
from .phi_calculator import PhiCalculator
from .system_analyzer import SystemAnalyzer
from .perturbation_engine import PerturbationEngine


class IITProcessor(BaseProcessor):
    """
    Integrated Information Theory processor for consciousness analysis.
    
    Implements IIT 3.0 concepts including:
    - Phi (Φ) calculation for integrated information
    - System partitioning and analysis
    - Cause-effect structure analysis
    - Perturbational complexity measurement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize sub-components
        self.phi_calculator = None
        self.system_analyzer = None
        self.perturbation_engine = None
        
        # Configuration
        self.phi_threshold = self.config.get('phi_threshold', 0.1)
        self.max_system_size = self.config.get('max_system_size', 10)
        self.perturbation_strength = self.config.get('perturbation_strength', 0.1)
        self.integration_steps = self.config.get('integration_steps', 1000)
        self.use_gaussian_perturbation = self.config.get('use_gaussian_perturbation', True)
        self.calculate_phi_max = self.config.get('calculate_phi_max', True)
    
    def initialize(self) -> bool:
        """Initialize IIT processor components."""
        try:
            self.phi_calculator = PhiCalculator(
                threshold=self.phi_threshold,
                max_system_size=self.max_system_size
            )
            
            self.system_analyzer = SystemAnalyzer(
                max_system_size=self.max_system_size
            )
            
            self.perturbation_engine = PerturbationEngine(
                strength=self.perturbation_strength,
                use_gaussian=self.use_gaussian_perturbation
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize IIT processor: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for IIT analysis."""
        if isinstance(data, dict):
            # Check for required fields
            if 'connectivity_matrix' in data:
                return DataValidator.validate_network_structure(data['connectivity_matrix'])
            elif 'timeseries' in data:
                return DataValidator.validate_neural_timeseries(data['timeseries'])
            elif 'system_state' in data:
                return isinstance(data['system_state'], np.ndarray) and data['system_state'].size > 0
        
        elif isinstance(data, np.ndarray):
            # Direct array input
            if len(data.shape) == 2:
                # Could be connectivity matrix or time series
                return DataValidator.validate_network_structure(data) or DataValidator.validate_neural_timeseries(data)
            elif len(data.shape) == 1:
                # Could be system state
                return data.size > 0
        
        return False
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """
        Process data using IIT analysis.
        
        Args:
            data: Input data (connectivity matrix, time series, or system state)
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult with IIT analysis results
        """
        try:
            # Parse input data
            parsed_data = self._parse_input_data(data)
            
            # Perform IIT analysis
            results = {}
            
            # Calculate integrated information (Phi)
            if 'system_state' in parsed_data:
                phi_results = self._calculate_phi(parsed_data['system_state'])
                results.update(phi_results)
            
            # Analyze system structure
            if 'connectivity_matrix' in parsed_data:
                structure_results = self._analyze_system_structure(parsed_data['connectivity_matrix'])
                results.update(structure_results)
            
            # Analyze time series dynamics
            if 'timeseries' in parsed_data:
                dynamics_results = self._analyze_dynamics(parsed_data['timeseries'])
                results.update(dynamics_results)
            
            # Calculate perturbational complexity
            if 'system_state' in parsed_data or 'timeseries' in parsed_data:
                perturbation_results = self._calculate_perturbational_complexity(parsed_data)
                results.update(perturbation_results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(results)
            
            return ProcessingResult(
                data=results,
                metadata={
                    'phi_threshold': self.phi_threshold,
                    'max_system_size': self.max_system_size,
                    'perturbation_strength': self.perturbation_strength,
                    'processing_parameters': kwargs
                },
                timestamp=datetime.now(),
                processor_type="IIT",
                confidence=confidence
            )
            
        except Exception as e:
            return ProcessingResult(
                data=None,
                metadata={'error_details': str(e)},
                timestamp=datetime.now(),
                processor_type="IIT",
                errors=[f"IIT processing failed: {str(e)}"]
            )
    
    def _parse_input_data(self, data: Any) -> Dict[str, np.ndarray]:
        """Parse and standardize input data."""
        parsed = {}
        
        if isinstance(data, dict):
            for key in ['connectivity_matrix', 'timeseries', 'system_state']:
                if key in data and isinstance(data[key], np.ndarray):
                    parsed[key] = data[key]
        
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 2:
                # Determine if it's connectivity matrix or time series
                if data.shape[0] == data.shape[1]:
                    parsed['connectivity_matrix'] = data
                else:
                    parsed['timeseries'] = data
            elif len(data.shape) == 1:
                parsed['system_state'] = data
        
        return parsed
    
    def _calculate_phi(self, system_state: np.ndarray) -> Dict[str, float]:
        """Calculate integrated information (Phi) for system state."""
        results = {}
        
        # Basic Phi calculation
        phi_value = self.phi_calculator.calculate_phi(system_state)
        results['phi'] = phi_value
        
        # Phi_max calculation if enabled
        if self.calculate_phi_max:
            phi_max = self.phi_calculator.calculate_phi_max(system_state)
            results['phi_max'] = phi_max
        
        # Normalized Phi
        results['phi_normalized'] = min(1.0, phi_value / (self.phi_threshold + 0.001))
        
        # Consciousness indicator based on Phi threshold
        results['consciousness_indicator'] = 1.0 if phi_value > self.phi_threshold else 0.0
        
        return results
    
    def _analyze_system_structure(self, connectivity_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze system structure and topology."""
        results = {}
        
        # Basic network metrics
        structure_metrics = self.system_analyzer.analyze_structure(connectivity_matrix)
        results.update(structure_metrics)
        
        # Identify main complex
        main_complex = self.system_analyzer.identify_main_complex(connectivity_matrix)
        results['main_complex'] = main_complex
        
        # System partitioning
        partitions = self.system_analyzer.find_optimal_partitions(connectivity_matrix)
        results['optimal_partitions'] = partitions
        
        return results
    
    def _analyze_dynamics(self, timeseries: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal dynamics for consciousness indicators."""
        results = {}
        
        # Calculate dynamic Phi over time
        if timeseries.shape[1] > 10:  # Need sufficient time points
            dynamic_phi = []
            window_size = min(10, timeseries.shape[1] // 4)
            
            for i in range(0, timeseries.shape[1] - window_size, window_size // 2):
                window_data = timeseries[:, i:i+window_size]
                # Use mean across time window as system state
                window_state = np.mean(window_data, axis=1)
                phi_val = self.phi_calculator.calculate_phi(window_state)
                dynamic_phi.append(phi_val)
            
            results['dynamic_phi'] = np.array(dynamic_phi)
            results['phi_variability'] = np.std(dynamic_phi)
            results['mean_phi'] = np.mean(dynamic_phi)
        
        # Temporal complexity
        results['temporal_complexity'] = self._calculate_temporal_complexity(timeseries)
        
        return results
    
    def _calculate_perturbational_complexity(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate perturbational complexity index (PCI)."""
        results = {}
        
        # Use system state or derive from time series
        if 'system_state' in data:
            base_state = data['system_state']
        elif 'timeseries' in data:
            base_state = np.mean(data['timeseries'], axis=1)
        else:
            return results
        
        # Generate perturbations and measure responses
        pci_score = self.perturbation_engine.calculate_pci(base_state)
        results['pci'] = pci_score
        
        # Normalized PCI
        results['pci_normalized'] = min(1.0, pci_score / 0.5)  # Typical PCI threshold
        
        return results
    
    def _calculate_temporal_complexity(self, timeseries: np.ndarray) -> float:
        """Calculate temporal complexity of the time series."""
        if timeseries.shape[1] < 2:
            return 0.0
        
        # Lempel-Ziv complexity on binarized time series
        complexity_scores = []
        
        for channel in timeseries:
            # Binarize relative to median
            binary_series = (channel > np.median(channel)).astype(int)
            
            # Calculate Lempel-Ziv complexity
            complexity = self._lempel_ziv_complexity(binary_series)
            complexity_scores.append(complexity)
        
        return np.mean(complexity_scores)
    
    def _lempel_ziv_complexity(self, binary_sequence: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity of binary sequence."""
        n = len(binary_sequence)
        if n == 0:
            return 0.0
        
        complexity = 0
        i = 0
        
        while i < n:
            k = 1
            while i + k <= n:
                subseq = binary_sequence[i:i+k]
                # Check if subsequence exists in previous part
                found = False
                for j in range(i):
                    if np.array_equal(subseq, binary_sequence[j:j+k]):
                        found = True
                        break
                
                if not found:
                    complexity += 1
                    break
                k += 1
            
            i += k if k > 1 else 1
        
        # Normalize by sequence length
        return complexity / n if n > 0 else 0.0
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score for IIT analysis."""
        confidence_factors = []
        
        # Phi-based confidence
        if 'phi' in results:
            phi_confidence = min(1.0, results['phi'] / (self.phi_threshold * 2))
            confidence_factors.append(phi_confidence)
        
        # PCI-based confidence
        if 'pci' in results:
            pci_confidence = min(1.0, results['pci'] / 0.3)
            confidence_factors.append(pci_confidence)
        
        # Structure-based confidence
        if 'integration' in results:
            structure_confidence = min(1.0, results['integration'])
            confidence_factors.append(structure_confidence)
        
        # Complexity-based confidence
        if 'temporal_complexity' in results:
            complexity_confidence = min(1.0, results['temporal_complexity'] * 2)
            confidence_factors.append(complexity_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5