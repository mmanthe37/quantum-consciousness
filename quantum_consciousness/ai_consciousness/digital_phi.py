"""
Digital Phi calculator for AI consciousness detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import entropy
from itertools import combinations


class DigitalPhiCalculator:
    """
    Calculator for digital Phi (Φ) in artificial neural networks.
    
    Adapts IIT concepts to digital systems by analyzing:
    - Information integration in network activations
    - Digital causation between network layers
    - Temporal integration in recurrent architectures
    """
    
    def __init__(self, resolution: int = 100):
        self.resolution = resolution
    
    def calculate_digital_phi(self, activations: np.ndarray) -> float:
        """
        Calculate digital Phi from neural network activations.
        
        Args:
            activations: Network activations (units x samples) or (layers x units x samples)
            
        Returns:
            Digital Phi value
        """
        if activations.size == 0:
            return 0.0
        
        # Ensure proper dimensionality
        if len(activations.shape) == 1:
            activations = activations.reshape(1, -1)
        elif len(activations.shape) > 2:
            # Flatten higher dimensions
            original_shape = activations.shape
            activations = activations.reshape(original_shape[0], -1)
        
        # Calculate system-level information
        system_info = self._calculate_system_information(activations)
        
        # Calculate minimum information partition
        mip_info = self._find_minimum_information_partition(activations)
        
        # Digital Phi is the difference
        digital_phi = max(0.0, system_info - mip_info)
        
        return digital_phi
    
    def calculate_temporal_phi(self, temporal_activations: np.ndarray) -> float:
        """
        Calculate temporal Phi for time-varying activations.
        
        Args:
            temporal_activations: Activations over time (units x time_steps)
            
        Returns:
            Temporal Phi value
        """
        if temporal_activations.size == 0 or len(temporal_activations.shape) < 2:
            return 0.0
        
        n_units, n_timesteps = temporal_activations.shape
        
        if n_timesteps < 2:
            return 0.0
        
        # Calculate temporal integration
        temporal_phi_values = []
        
        # Use sliding window approach
        window_size = min(10, n_timesteps // 2)
        
        for t in range(n_timesteps - window_size + 1):
            window_activations = temporal_activations[:, t:t + window_size]
            
            # Calculate Phi for this temporal window
            window_phi = self.calculate_digital_phi(window_activations)
            temporal_phi_values.append(window_phi)
        
        # Temporal Phi is the mean across windows
        return np.mean(temporal_phi_values) if temporal_phi_values else 0.0
    
    def calculate_layer_phi(self, layer_activations: np.ndarray) -> List[float]:
        """
        Calculate Phi for each layer in a multi-layer network.
        
        Args:
            layer_activations: Activations (layers x units x samples)
            
        Returns:
            List of Phi values for each layer
        """
        if len(layer_activations.shape) < 3:
            return [self.calculate_digital_phi(layer_activations)]
        
        layer_phi_values = []
        
        for layer_idx in range(layer_activations.shape[0]):
            layer_data = layer_activations[layer_idx]
            layer_phi = self.calculate_digital_phi(layer_data)
            layer_phi_values.append(layer_phi)
        
        return layer_phi_values
    
    def calculate_cross_layer_phi(self, layer_activations: np.ndarray) -> float:
        """
        Calculate Phi across multiple layers (inter-layer integration).
        
        Args:
            layer_activations: Activations (layers x units x samples)
            
        Returns:
            Cross-layer Phi value
        """
        if len(layer_activations.shape) < 3 or layer_activations.shape[0] < 2:
            return 0.0
        
        n_layers = layer_activations.shape[0]
        
        # Calculate pairwise layer interactions
        cross_layer_interactions = []
        
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                layer_i = layer_activations[i]
                layer_j = layer_activations[j]
                
                # Calculate mutual information between layers
                mutual_info = self._calculate_mutual_information(layer_i, layer_j)
                cross_layer_interactions.append(mutual_info)
        
        # Cross-layer Phi as mean interaction strength
        return np.mean(cross_layer_interactions) if cross_layer_interactions else 0.0
    
    def calculate_digital_phi_spectrum(self, 
                                     activations: np.ndarray,
                                     scales: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Calculate digital Phi across multiple scales.
        
        Args:
            activations: Network activations
            scales: List of spatial scales to analyze
            
        Returns:
            Dictionary mapping scales to Phi values
        """
        if activations.size == 0:
            return {}
        
        # Default scales based on activation dimensions
        if scales is None:
            max_scale = min(8, activations.shape[0] // 2) if len(activations.shape) > 1 else 4
            scales = [2**i for i in range(int(np.log2(max_scale)) + 1) if 2**i <= max_scale]
        
        phi_spectrum = {}
        
        for scale in scales:
            if scale >= activations.shape[0]:
                continue
            
            # Coarse-grain activations at this scale
            coarse_grained = self._coarse_grain_activations(activations, scale)
            
            # Calculate Phi at this scale
            phi_spectrum[scale] = self.calculate_digital_phi(coarse_grained)
        
        return phi_spectrum
    
    def _calculate_system_information(self, activations: np.ndarray) -> float:
        """Calculate information content of the entire system."""
        if activations.size == 0:
            return 0.0
        
        # Discretize activations for entropy calculation
        discretized = self._discretize_activations(activations)
        
        # Calculate joint entropy
        system_entropy = self._calculate_joint_entropy(discretized)
        
        return system_entropy
    
    def _find_minimum_information_partition(self, activations: np.ndarray) -> float:
        """Find the partition that minimizes information."""
        n_units = activations.shape[0]
        
        if n_units <= 1:
            return self._calculate_system_information(activations)
        
        min_info = float('inf')
        
        # Try all possible bipartitions
        for partition_size in range(1, n_units):
            for partition in combinations(range(n_units), partition_size):
                partition_info = self._calculate_partition_information(activations, partition)
                min_info = min(min_info, partition_info)
        
        return min_info
    
    def _calculate_partition_information(self, 
                                       activations: np.ndarray, 
                                       partition: Tuple[int, ...]) -> float:
        """Calculate information for a specific partition."""
        # Split activations into two parts
        partition_indices = list(partition)
        complement_indices = [i for i in range(activations.shape[0]) if i not in partition_indices]
        
        part1 = activations[partition_indices] if partition_indices else np.array([])
        part2 = activations[complement_indices] if complement_indices else np.array([])
        
        # Calculate information for each part
        info1 = self._calculate_system_information(part1) if part1.size > 0 else 0
        info2 = self._calculate_system_information(part2) if part2.size > 0 else 0
        
        # Weighted sum
        total_size = activations.shape[0]
        weight1 = len(partition_indices) / total_size if total_size > 0 else 0
        weight2 = len(complement_indices) / total_size if total_size > 0 else 0
        
        return weight1 * info1 + weight2 * info2
    
    def _discretize_activations(self, activations: np.ndarray) -> np.ndarray:
        """Discretize continuous activations for entropy calculation."""
        if activations.size == 0:
            return activations
        
        # Use quantile-based discretization
        n_bins = min(self.resolution, max(10, int(np.sqrt(activations.shape[1]))))
        
        discretized = np.zeros_like(activations, dtype=int)
        
        for unit_idx in range(activations.shape[0]):
            unit_activations = activations[unit_idx]
            
            # Handle constant activations
            if np.std(unit_activations) < 1e-10:
                discretized[unit_idx] = 0
                continue
            
            # Quantile-based binning
            try:
                _, bin_edges = np.histogram(unit_activations, bins=n_bins)
                discretized[unit_idx] = np.digitize(unit_activations, bin_edges) - 1
                discretized[unit_idx] = np.clip(discretized[unit_idx], 0, n_bins - 1)
            except:
                discretized[unit_idx] = 0
        
        return discretized
    
    def _calculate_joint_entropy(self, discretized_activations: np.ndarray) -> float:
        """Calculate joint entropy of discretized activations."""
        if discretized_activations.size == 0:
            return 0.0
        
        # For computational efficiency, limit to reasonable number of units
        max_units = min(10, discretized_activations.shape[0])
        
        if discretized_activations.shape[0] > max_units:
            # Sample units randomly
            selected_units = np.random.choice(
                discretized_activations.shape[0], 
                max_units, 
                replace=False
            )
            activations_subset = discretized_activations[selected_units]
        else:
            activations_subset = discretized_activations
        
        # Calculate joint distribution
        try:
            # Convert to tuple format for counting
            joint_states = []
            for sample_idx in range(activations_subset.shape[1]):
                state_tuple = tuple(activations_subset[:, sample_idx])
                joint_states.append(state_tuple)
            
            # Count occurrences
            unique_states, counts = np.unique(joint_states, return_counts=True, axis=0)
            probabilities = counts / np.sum(counts)
            
            # Calculate entropy
            joint_entropy = entropy(probabilities, base=2)
            
            return joint_entropy
            
        except:
            # Fallback: sum of individual entropies (independence assumption)
            individual_entropies = []
            for unit_idx in range(activations_subset.shape[0]):
                unit_data = activations_subset[unit_idx]
                unique_vals, counts = np.unique(unit_data, return_counts=True)
                probs = counts / np.sum(counts)
                unit_entropy = entropy(probs, base=2)
                individual_entropies.append(unit_entropy)
            
            return np.sum(individual_entropies)
    
    def _calculate_mutual_information(self, 
                                    activations1: np.ndarray, 
                                    activations2: np.ndarray) -> float:
        """Calculate mutual information between two sets of activations."""
        if activations1.size == 0 or activations2.size == 0:
            return 0.0
        
        # Ensure same number of samples
        min_samples = min(activations1.shape[-1], activations2.shape[-1])
        act1 = activations1[..., :min_samples]
        act2 = activations2[..., :min_samples]
        
        # Discretize both sets
        disc1 = self._discretize_activations(act1)
        disc2 = self._discretize_activations(act2)
        
        # Calculate entropies
        h1 = self._calculate_joint_entropy(disc1)
        h2 = self._calculate_joint_entropy(disc2)
        
        # Joint entropy
        combined = np.vstack([disc1, disc2])
        h_joint = self._calculate_joint_entropy(combined)
        
        # Mutual information
        mutual_info = h1 + h2 - h_joint
        
        return max(0.0, mutual_info)
    
    def _coarse_grain_activations(self, activations: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain activations by averaging over spatial groups."""
        if scale >= activations.shape[0]:
            return np.mean(activations, axis=0, keepdims=True)
        
        n_groups = activations.shape[0] // scale
        coarse_grained = np.zeros((n_groups, activations.shape[1]))
        
        for group_idx in range(n_groups):
            start_idx = group_idx * scale
            end_idx = min(start_idx + scale, activations.shape[0])
            
            group_activations = activations[start_idx:end_idx]
            coarse_grained[group_idx] = np.mean(group_activations, axis=0)
        
        return coarse_grained
    
    def calculate_causal_phi(self, 
                           pre_activations: np.ndarray,
                           post_activations: np.ndarray) -> float:
        """
        Calculate causal Phi between pre and post states.
        
        Args:
            pre_activations: Activations before processing
            post_activations: Activations after processing
            
        Returns:
            Causal Phi value
        """
        if pre_activations.size == 0 or post_activations.size == 0:
            return 0.0
        
        # Calculate causal information
        causal_info = self._calculate_mutual_information(pre_activations, post_activations)
        
        # Calculate noise-corrected causal strength
        # This is a simplified version of more complex causal analysis
        noise_level = self._estimate_noise_level(pre_activations, post_activations)
        
        # Causal Phi accounts for both information transfer and noise
        causal_phi = causal_info * (1.0 - noise_level)
        
        return max(0.0, causal_phi)
    
    def _estimate_noise_level(self, 
                            pre_activations: np.ndarray,
                            post_activations: np.ndarray) -> float:
        """Estimate noise level in causal relationship."""
        try:
            # Calculate correlation structure
            if pre_activations.shape != post_activations.shape:
                return 0.5  # High uncertainty
            
            # Element-wise correlation
            correlations = []
            for i in range(min(10, pre_activations.shape[0])):  # Limit for efficiency
                pre_unit = pre_activations[i]
                post_unit = post_activations[i]
                
                if np.std(pre_unit) > 1e-10 and np.std(post_unit) > 1e-10:
                    corr = np.corrcoef(pre_unit, post_unit)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                mean_correlation = np.mean(correlations)
                noise_level = 1.0 - mean_correlation  # Inverse of correlation
                return np.clip(noise_level, 0.0, 1.0)
            else:
                return 0.5
                
        except:
            return 0.5
    
    def calculate_feedforward_phi(self, layer_sequence: List[np.ndarray]) -> List[float]:
        """
        Calculate Phi for feedforward information flow.
        
        Args:
            layer_sequence: List of layer activations in feedforward order
            
        Returns:
            List of Phi values for each layer transition
        """
        if len(layer_sequence) < 2:
            return []
        
        phi_values = []
        
        for i in range(len(layer_sequence) - 1):
            pre_layer = layer_sequence[i]
            post_layer = layer_sequence[i + 1]
            
            # Calculate causal Phi between consecutive layers
            causal_phi = self.calculate_causal_phi(pre_layer, post_layer)
            phi_values.append(causal_phi)
        
        return phi_values