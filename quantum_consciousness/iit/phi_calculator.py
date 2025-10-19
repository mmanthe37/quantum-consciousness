"""
Phi (Φ) calculator for Integrated Information Theory.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import combinations
from scipy.stats import entropy
from scipy.optimize import minimize


class PhiCalculator:
    """
    Calculator for integrated information (Phi) measures in IIT.
    
    Implements various Phi calculation methods including:
    - Basic Phi calculation
    - Phi_max calculation
    - Cause-effect structure analysis
    """
    
    def __init__(self, threshold: float = 0.1, max_system_size: int = 10):
        self.threshold = threshold
        self.max_system_size = max_system_size
    
    def calculate_phi(self, system_state: np.ndarray) -> float:
        """
        Calculate basic integrated information (Phi) for a system state.
        
        Args:
            system_state: System state vector
            
        Returns:
            Phi value (integrated information)
        """
        if system_state.size == 0 or system_state.size > self.max_system_size:
            return 0.0
        
        # Normalize system state
        state_norm = self._normalize_state(system_state)
        
        # Calculate system entropy
        system_entropy = self._calculate_entropy(state_norm)
        
        # Calculate sum of partition entropies for minimum information partition (MIP)
        mip_entropy = self._find_minimum_information_partition(state_norm)
        
        # Phi is the difference
        phi = max(0.0, system_entropy - mip_entropy)
        
        return phi
    
    def calculate_phi_max(self, system_state: np.ndarray) -> float:
        """
        Calculate Phi_max - the maximum Phi over all possible system cuts.
        
        Args:
            system_state: System state vector
            
        Returns:
            Maximum Phi value
        """
        if system_state.size <= 1:
            return 0.0
        
        max_phi = 0.0
        n_elements = len(system_state)
        
        # Try all possible bipartitions
        for partition_size in range(1, n_elements):
            for partition in combinations(range(n_elements), partition_size):
                phi_value = self._calculate_phi_for_partition(system_state, partition)
                max_phi = max(max_phi, phi_value)
        
        return max_phi
    
    def calculate_cause_effect_structure(self, system_state: np.ndarray) -> Dict[str, float]:
        """
        Calculate cause-effect structure measures.
        
        Args:
            system_state: System state vector
            
        Returns:
            Dictionary with cause-effect structure metrics
        """
        if system_state.size <= 1:
            return {"cause_information": 0.0, "effect_information": 0.0, "ces_phi": 0.0}
        
        # Calculate cause and effect information
        cause_info = self._calculate_cause_information(system_state)
        effect_info = self._calculate_effect_information(system_state)
        
        # CES Phi is minimum of cause and effect information
        ces_phi = min(cause_info, effect_info)
        
        return {
            "cause_information": cause_info,
            "effect_information": effect_info,
            "ces_phi": ces_phi
        }
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize system state to probability distribution."""
        state_abs = np.abs(state)
        total = np.sum(state_abs)
        
        if total == 0:
            return np.ones(len(state)) / len(state)
        
        return state_abs / total
    
    def _calculate_entropy(self, prob_dist: np.ndarray) -> float:
        """Calculate entropy of probability distribution."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        prob_dist_safe = prob_dist + epsilon
        return -np.sum(prob_dist_safe * np.log2(prob_dist_safe))
    
    def _find_minimum_information_partition(self, state: np.ndarray) -> float:
        """Find minimum information partition (MIP) entropy."""
        if len(state) <= 1:
            return self._calculate_entropy(state)
        
        min_entropy = float('inf')
        n_elements = len(state)
        
        # Try all possible bipartitions
        for partition_size in range(1, n_elements):
            for partition in combinations(range(n_elements), partition_size):
                partition_entropy = self._calculate_partition_entropy(state, partition)
                min_entropy = min(min_entropy, partition_entropy)
        
        return min_entropy
    
    def _calculate_partition_entropy(self, state: np.ndarray, partition: Tuple[int, ...]) -> float:
        """Calculate entropy for a specific partition."""
        # Create partition masks
        partition_mask = np.zeros(len(state), dtype=bool)
        partition_mask[list(partition)] = True
        
        # Split state into two parts
        part1 = state[partition_mask]
        part2 = state[~partition_mask]
        
        # Calculate individual entropies
        entropy1 = self._calculate_entropy(part1) if len(part1) > 0 else 0
        entropy2 = self._calculate_entropy(part2) if len(part2) > 0 else 0
        
        # Weighted sum based on partition sizes
        weight1 = len(part1) / len(state)
        weight2 = len(part2) / len(state)
        
        return weight1 * entropy1 + weight2 * entropy2
    
    def _calculate_phi_for_partition(self, state: np.ndarray, partition: Tuple[int, ...]) -> float:
        """Calculate Phi for a specific partition."""
        # This is a simplified version - full IIT calculation is more complex
        whole_entropy = self._calculate_entropy(self._normalize_state(state))
        partition_entropy = self._calculate_partition_entropy(self._normalize_state(state), partition)
        
        return max(0.0, whole_entropy - partition_entropy)
    
    def _calculate_cause_information(self, state: np.ndarray) -> float:
        """Calculate cause information for the system."""
        # Simplified cause information calculation
        # In full IIT, this involves analyzing causal mechanisms
        
        if len(state) <= 1:
            return 0.0
        
        # Use temporal correlation as proxy for causal information
        cause_strength = 0.0
        
        for i in range(len(state)):
            for j in range(len(state)):
                if i != j:
                    # Simple correlation measure
                    correlation = abs(state[i] * state[j]) / (np.sqrt(state[i]**2 + 1e-10) * np.sqrt(state[j]**2 + 1e-10))
                    cause_strength += correlation
        
        # Normalize by number of connections
        n_connections = len(state) * (len(state) - 1)
        return cause_strength / n_connections if n_connections > 0 else 0.0
    
    def _calculate_effect_information(self, state: np.ndarray) -> float:
        """Calculate effect information for the system."""
        # Simplified effect information calculation
        # In full IIT, this involves analyzing effect mechanisms
        
        if len(state) <= 1:
            return 0.0
        
        # Use state coherence as proxy for effect information
        mean_state = np.mean(state)
        variance = np.var(state)
        
        # Effect information based on how coherently the system responds
        if variance < 1e-10:
            return 1.0  # Perfect coherence
        
        coherence = 1.0 / (1.0 + variance)
        return min(1.0, coherence)
    
    def calculate_phi_composition(self, system_state: np.ndarray) -> Dict[str, float]:
        """
        Calculate compositional Phi measures.
        
        Args:
            system_state: System state vector
            
        Returns:
            Dictionary with compositional Phi metrics
        """
        if system_state.size <= 2:
            return {"phi_comp": 0.0, "phi_integration": 0.0, "phi_differentiation": 0.0}
        
        # Calculate integration component
        phi_integration = self._calculate_integration_phi(system_state)
        
        # Calculate differentiation component
        phi_differentiation = self._calculate_differentiation_phi(system_state)
        
        # Compositional Phi combines both
        phi_comp = (phi_integration * phi_differentiation) ** 0.5
        
        return {
            "phi_comp": phi_comp,
            "phi_integration": phi_integration,
            "phi_differentiation": phi_differentiation
        }
    
    def _calculate_integration_phi(self, state: np.ndarray) -> float:
        """Calculate integration component of Phi."""
        # Measure how much information is lost when system is partitioned
        whole_info = self._calculate_entropy(self._normalize_state(state))
        
        # Find maximum information partition
        max_partition_info = 0.0
        n_elements = len(state)
        
        for partition_size in range(1, n_elements):
            for partition in combinations(range(n_elements), partition_size):
                partition_info = self._calculate_partition_entropy(self._normalize_state(state), partition)
                max_partition_info = max(max_partition_info, partition_info)
        
        return max(0.0, whole_info - max_partition_info)
    
    def _calculate_differentiation_phi(self, state: np.ndarray) -> float:
        """Calculate differentiation component of Phi."""
        # Measure how much the system can be differentiated from noise
        if len(state) <= 1:
            return 0.0
        
        # Calculate signal-to-noise ratio as differentiation proxy
        signal_power = np.var(state)
        noise_floor = np.mean(np.abs(state)) * 0.1  # Assume 10% noise floor
        
        if noise_floor < 1e-10:
            return 1.0
        
        snr = signal_power / noise_floor
        differentiation = min(1.0, np.log(1 + snr) / np.log(10))  # Log scale normalization
        
        return differentiation
    
    def calculate_phi_spectrum(self, system_state: np.ndarray, scales: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Calculate Phi across multiple scales.
        
        Args:
            system_state: System state vector
            scales: List of scales to analyze (default: [1, 2, 4, 8])
            
        Returns:
            Dictionary mapping scales to Phi values
        """
        if scales is None:
            max_scale = min(8, len(system_state) // 2)
            scales = [2**i for i in range(int(np.log2(max_scale)) + 1) if 2**i <= max_scale]
        
        phi_spectrum = {}
        
        for scale in scales:
            if scale >= len(system_state):
                continue
                
            # Coarse-grain the system at this scale
            coarse_grained = self._coarse_grain_state(system_state, scale)
            
            # Calculate Phi at this scale
            phi_spectrum[scale] = self.calculate_phi(coarse_grained)
        
        return phi_spectrum
    
    def _coarse_grain_state(self, state: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain system state at specified scale."""
        if scale >= len(state):
            return np.array([np.mean(state)])
        
        n_groups = len(state) // scale
        coarse_grained = np.zeros(n_groups)
        
        for i in range(n_groups):
            start_idx = i * scale
            end_idx = min(start_idx + scale, len(state))
            coarse_grained[i] = np.mean(state[start_idx:end_idx])
        
        return coarse_grained