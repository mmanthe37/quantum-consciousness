"""
Perturbation engine for measuring perturbational complexity index (PCI).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.signal import hilbert
from scipy.stats import zscore


class PerturbationEngine:
    """
    Engine for generating perturbations and measuring system responses.
    
    Implements perturbational complexity index (PCI) calculation
    for consciousness assessment.
    """
    
    def __init__(self, strength: float = 0.1, use_gaussian: bool = True):
        self.strength = strength
        self.use_gaussian = use_gaussian
    
    def calculate_pci(self, system_state: np.ndarray, n_perturbations: int = 100) -> float:
        """
        Calculate Perturbational Complexity Index (PCI).
        
        Args:
            system_state: Base system state
            n_perturbations: Number of perturbations to generate
            
        Returns:
            PCI score
        """
        if system_state.size == 0:
            return 0.0
        
        # Generate perturbations and measure responses
        responses = []
        
        for _ in range(n_perturbations):
            # Generate perturbation
            perturbation = self._generate_perturbation(system_state.shape)
            
            # Apply perturbation
            perturbed_state = system_state + perturbation
            
            # Calculate response
            response = self._calculate_response(system_state, perturbed_state)
            responses.append(response)
        
        # Calculate PCI from responses
        pci_score = self._calculate_pci_from_responses(responses)
        
        return pci_score
    
    def calculate_spatiotemporal_pci(self, 
                                   timeseries: np.ndarray, 
                                   perturbation_timepoints: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Calculate spatiotemporal PCI from time series data.
        
        Args:
            timeseries: Time series data (channels x time)
            perturbation_timepoints: Specific timepoints to perturb
            
        Returns:
            Dictionary with PCI metrics
        """
        if timeseries.size == 0:
            return {"pci": 0.0, "spatial_pci": 0.0, "temporal_pci": 0.0}
        
        n_channels, n_timepoints = timeseries.shape
        
        # Default perturbation timepoints
        if perturbation_timepoints is None:
            perturbation_timepoints = np.linspace(
                n_timepoints // 4, 
                3 * n_timepoints // 4, 
                10, 
                dtype=int
            ).tolist()
        
        spatial_responses = []
        temporal_responses = []
        
        for t_pert in perturbation_timepoints:
            if t_pert >= n_timepoints:
                continue
            
            # Generate spatial perturbation at timepoint
            spatial_perturbation = self._generate_perturbation((n_channels,))
            
            # Create perturbed timeseries
            perturbed_ts = timeseries.copy()
            perturbed_ts[:, t_pert] += spatial_perturbation
            
            # Calculate spatial response (across channels)
            spatial_response = self._calculate_spatial_response(timeseries, perturbed_ts, t_pert)
            spatial_responses.append(spatial_response)
            
            # Calculate temporal response (across time)
            temporal_response = self._calculate_temporal_response(timeseries, perturbed_ts, t_pert)
            temporal_responses.append(temporal_response)
        
        # Calculate PCI scores
        spatial_pci = self._calculate_pci_from_responses(spatial_responses)
        temporal_pci = self._calculate_pci_from_responses(temporal_responses)
        
        # Combined PCI
        combined_pci = (spatial_pci * temporal_pci) ** 0.5
        
        return {
            "pci": combined_pci,
            "spatial_pci": spatial_pci,
            "temporal_pci": temporal_pci
        }
    
    def measure_causal_emergence(self, 
                                system_state: np.ndarray,
                                connectivity_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Measure causal emergence through perturbational analysis.
        
        Args:
            system_state: System state vector
            connectivity_matrix: Optional connectivity structure
            
        Returns:
            Dictionary with causal emergence metrics
        """
        if system_state.size == 0:
            return {"causal_emergence": 0.0, "downward_causation": 0.0}
        
        # Measure bottom-up causation
        bottom_up_influence = self._measure_bottom_up_causation(system_state)
        
        # Measure top-down causation
        top_down_influence = self._measure_top_down_causation(system_state)
        
        # Calculate causal emergence
        if bottom_up_influence > 0:
            causal_emergence = top_down_influence / (bottom_up_influence + 1e-10)
        else:
            causal_emergence = 0.0
        
        return {
            "causal_emergence": causal_emergence,
            "downward_causation": top_down_influence,
            "upward_causation": bottom_up_influence
        }
    
    def _generate_perturbation(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate perturbation with specified characteristics."""
        if self.use_gaussian:
            perturbation = np.random.normal(0, self.strength, shape)
        else:
            # Uniform perturbation
            perturbation = np.random.uniform(-self.strength, self.strength, shape)
        
        return perturbation
    
    def _calculate_response(self, original_state: np.ndarray, perturbed_state: np.ndarray) -> np.ndarray:
        """Calculate system response to perturbation."""
        # Simple response: difference between states
        response = perturbed_state - original_state
        
        # Normalize response
        response_norm = np.linalg.norm(response)
        if response_norm > 0:
            response = response / response_norm
        
        return response
    
    def _calculate_spatial_response(self, 
                                  original_ts: np.ndarray, 
                                  perturbed_ts: np.ndarray, 
                                  perturbation_time: int) -> np.ndarray:
        """Calculate spatial response to perturbation."""
        # Look at response in time window after perturbation
        window_size = min(10, original_ts.shape[1] - perturbation_time)
        
        if window_size <= 0:
            return np.zeros(original_ts.shape[0])
        
        # Calculate mean response in post-perturbation window
        post_pert_orig = original_ts[:, perturbation_time:perturbation_time + window_size]
        post_pert_perturbed = perturbed_ts[:, perturbation_time:perturbation_time + window_size]
        
        spatial_response = np.mean(post_pert_perturbed - post_pert_orig, axis=1)
        
        return spatial_response
    
    def _calculate_temporal_response(self, 
                                   original_ts: np.ndarray, 
                                   perturbed_ts: np.ndarray, 
                                   perturbation_time: int) -> np.ndarray:
        """Calculate temporal response to perturbation."""
        # Calculate response as temporal evolution of perturbation
        diff_ts = perturbed_ts - original_ts
        
        # Focus on post-perturbation period
        if perturbation_time < diff_ts.shape[1] - 1:
            temporal_response = np.mean(diff_ts[:, perturbation_time:], axis=0)
        else:
            temporal_response = np.mean(diff_ts, axis=0)
        
        return temporal_response
    
    def _calculate_pci_from_responses(self, responses: List[np.ndarray]) -> float:
        """Calculate PCI score from response patterns."""
        if not responses:
            return 0.0
        
        # Stack responses
        response_matrix = np.array([resp.flatten() for resp in responses])
        
        if response_matrix.size == 0:
            return 0.0
        
        # Calculate Lempel-Ziv complexity of response patterns
        pci_scores = []
        
        for i in range(response_matrix.shape[1]):  # For each response dimension
            response_series = response_matrix[:, i]
            
            # Binarize response series
            threshold = np.median(response_series)
            binary_series = (response_series > threshold).astype(int)
            
            # Calculate Lempel-Ziv complexity
            lz_complexity = self._lempel_ziv_complexity(binary_series)
            pci_scores.append(lz_complexity)
        
        # Return mean PCI across dimensions
        return np.mean(pci_scores) if pci_scores else 0.0
    
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
                
                # Check if this subsequence appeared before
                found = False
                for j in range(i):
                    if j + k <= len(binary_sequence):
                        if np.array_equal(subseq, binary_sequence[j:j+k]):
                            found = True
                            break
                
                if not found:
                    complexity += 1
                    break
                k += 1
                
                # Prevent infinite loop
                if k > n - i:
                    complexity += 1
                    break
            
            i += max(1, k)
        
        # Normalize by theoretical maximum
        theoretical_max = n / np.log2(n) if n > 1 else 1
        return complexity / theoretical_max
    
    def _measure_bottom_up_causation(self, system_state: np.ndarray) -> float:
        """Measure bottom-up causal influence."""
        n_elements = len(system_state)
        
        if n_elements <= 1:
            return 0.0
        
        # Perturb individual elements and measure system response
        individual_influences = []
        
        for i in range(n_elements):
            # Create perturbation for single element
            perturbation = np.zeros_like(system_state)
            perturbation[i] = self.strength
            
            # Calculate system response
            perturbed_state = system_state + perturbation
            system_response = np.linalg.norm(perturbed_state - system_state)
            
            individual_influences.append(system_response)
        
        # Bottom-up causation is sum of individual influences
        return np.sum(individual_influences)
    
    def _measure_top_down_causation(self, system_state: np.ndarray) -> float:
        """Measure top-down causal influence."""
        n_elements = len(system_state)
        
        if n_elements <= 1:
            return 0.0
        
        # Perturb system as a whole and measure individual responses
        system_perturbation = self._generate_perturbation(system_state.shape)
        perturbed_state = system_state + system_perturbation
        
        # Measure how individual elements respond to system perturbation
        element_responses = []
        
        for i in range(n_elements):
            element_response = abs(perturbed_state[i] - system_state[i])
            element_responses.append(element_response)
        
        # Top-down causation is variance in individual responses
        # (indicates differential response to global perturbation)
        if len(element_responses) > 1:
            return np.var(element_responses)
        else:
            return 0.0
    
    def generate_perturbation_protocol(self, 
                                     system_size: int,
                                     protocol_type: str = "systematic") -> List[Dict[str, Any]]:
        """
        Generate a perturbation protocol for systematic testing.
        
        Args:
            system_size: Size of the system to perturb
            protocol_type: Type of protocol ("systematic", "random", "targeted")
            
        Returns:
            List of perturbation specifications
        """
        protocol = []
        
        if protocol_type == "systematic":
            # Systematic perturbation of each element
            for i in range(system_size):
                perturbation_spec = {
                    "type": "single_element",
                    "target": i,
                    "strength": self.strength,
                    "duration": 1
                }
                protocol.append(perturbation_spec)
            
            # Add some combination perturbations
            if system_size > 2:
                for i in range(0, system_size - 1, 2):
                    perturbation_spec = {
                        "type": "dual_element",
                        "targets": [i, min(i + 1, system_size - 1)],
                        "strength": self.strength * 0.7,
                        "duration": 1
                    }
                    protocol.append(perturbation_spec)
        
        elif protocol_type == "random":
            # Random perturbations
            n_perturbations = min(20, system_size * 2)
            
            for _ in range(n_perturbations):
                n_targets = np.random.randint(1, min(system_size, 4) + 1)
                targets = np.random.choice(system_size, n_targets, replace=False).tolist()
                
                perturbation_spec = {
                    "type": "multi_element",
                    "targets": targets,
                    "strength": np.random.uniform(self.strength * 0.5, self.strength * 1.5),
                    "duration": np.random.randint(1, 4)
                }
                protocol.append(perturbation_spec)
        
        elif protocol_type == "targeted":
            # Target based on system structure (simplified)
            # This would ideally use connectivity information
            
            # Target central elements
            central_elements = [system_size // 2] if system_size > 1 else [0]
            
            for target in central_elements:
                perturbation_spec = {
                    "type": "central_hub",
                    "target": target,
                    "strength": self.strength * 1.2,
                    "duration": 2
                }
                protocol.append(perturbation_spec)
            
            # Target peripheral elements
            peripheral = [0, system_size - 1] if system_size > 2 else []
            
            for target in peripheral:
                perturbation_spec = {
                    "type": "peripheral",
                    "target": target,
                    "strength": self.strength * 0.8,
                    "duration": 1
                }
                protocol.append(perturbation_spec)
        
        return protocol