"""
Complexity analyzer for neural networks and AI systems.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from scipy.stats import entropy
from scipy.signal import welch


class ComplexityAnalyzer:
    """
    Analyzer for complexity measures in neural networks.
    
    Analyzes various forms of complexity including:
    - Perturbational complexity (Neural PCI)
    - Parameter complexity
    - Computational complexity
    - Lempel-Ziv complexity
    """
    
    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps
    
    def calculate_neural_pci(self, activations: np.ndarray, perturbation_strength: float = 0.1) -> float:
        """
        Calculate Neural Perturbational Complexity Index.
        
        Args:
            activations: Neural network activations
            perturbation_strength: Strength of perturbations
            
        Returns:
            Neural PCI score
        """
        if activations.size == 0:
            return 0.0
        
        # Generate perturbations
        base_state = np.mean(activations, axis=-1) if len(activations.shape) > 1 else activations
        
        responses = []
        n_perturbations = min(self.n_steps, 100)  # Limit for efficiency
        
        for _ in range(n_perturbations):
            # Generate random perturbation
            perturbation = np.random.normal(0, perturbation_strength, base_state.shape)
            
            # Apply perturbation
            perturbed_state = base_state + perturbation
            
            # Calculate response (simplified - would need actual network forward pass)
            response = self._simulate_network_response(base_state, perturbed_state)
            responses.append(response)
        
        # Calculate PCI from responses
        pci_score = self._calculate_pci_from_responses(responses)
        
        return pci_score
    
    def analyze_parameter_complexity(self, model_parameters: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze complexity of model parameters.
        
        Args:
            model_parameters: Dictionary of model parameters
            
        Returns:
            Dictionary with parameter complexity metrics
        """
        results = {}
        
        if not model_parameters:
            return results
        
        # Combine all parameters
        all_params = []
        for param_name, param_values in model_parameters.items():
            if isinstance(param_values, np.ndarray) and param_values.size > 0:
                all_params.extend(param_values.flatten())
        
        if not all_params:
            return results
        
        all_params = np.array(all_params)
        
        # Parameter distribution complexity
        results['parameter_entropy'] = self._calculate_parameter_entropy(all_params)
        
        # Parameter variance (diversity)
        results['parameter_variance'] = np.var(all_params)
        
        # Parameter sparsity
        results['parameter_sparsity'] = self._calculate_parameter_sparsity(all_params)
        
        # Parameter scale complexity
        results['parameter_scale_complexity'] = self._calculate_scale_complexity(all_params)
        
        # Layer-wise complexity if structured
        if len(model_parameters) > 1:
            results['layer_complexity_variance'] = self._calculate_layer_complexity_variance(model_parameters)
        
        return results
    
    def calculate_computational_complexity(self, activations: np.ndarray) -> float:
        """
        Calculate computational complexity from activations.
        
        Args:
            activations: Network activations
            
        Returns:
            Computational complexity score
        """
        if activations.size == 0:
            return 0.0
        
        # Ensure 2D shape (units x samples)
        if len(activations.shape) == 1:
            activations = activations.reshape(-1, 1)
        elif len(activations.shape) > 2:
            original_shape = activations.shape
            activations = activations.reshape(original_shape[0], -1)
        
        complexity_measures = []
        
        # Lempel-Ziv complexity
        lz_complexity = self._calculate_lempel_ziv_complexity(activations)
        complexity_measures.append(lz_complexity)
        
        # Spectral complexity
        spectral_complexity = self._calculate_spectral_complexity(activations)
        complexity_measures.append(spectral_complexity)
        
        # Correlation complexity
        correlation_complexity = self._calculate_correlation_complexity(activations)
        complexity_measures.append(correlation_complexity)
        
        # Return average complexity
        return np.mean(complexity_measures) if complexity_measures else 0.0
    
    def calculate_information_complexity(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate various information-theoretic complexity measures.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with information complexity metrics
        """
        results = {}
        
        if data.size == 0:
            return results
        
        # Flatten data if needed
        flat_data = data.flatten()
        
        # Shannon entropy
        results['shannon_entropy'] = self._calculate_shannon_entropy(flat_data)
        
        # Kolmogorov complexity approximation
        results['kolmogorov_approximation'] = self._approximate_kolmogorov_complexity(flat_data)
        
        # Logical depth approximation
        results['logical_depth'] = self._approximate_logical_depth(flat_data)
        
        # Effective complexity
        results['effective_complexity'] = self._calculate_effective_complexity(flat_data)
        
        return results
    
    def analyze_temporal_complexity(self, temporal_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze temporal complexity patterns.
        
        Args:
            temporal_data: Temporal data (features x time)
            
        Returns:
            Dictionary with temporal complexity metrics
        """
        results = {}
        
        if temporal_data.size == 0 or len(temporal_data.shape) < 2:
            return results
        
        # Temporal Lempel-Ziv complexity
        results['temporal_lz_complexity'] = self._calculate_temporal_lz_complexity(temporal_data)
        
        # Temporal correlation dimension
        results['temporal_correlation_dimension'] = self._calculate_temporal_correlation_dimension(temporal_data)
        
        # Temporal predictability
        results['temporal_predictability'] = self._calculate_temporal_predictability(temporal_data)
        
        # Temporal entropy rate
        results['temporal_entropy_rate'] = self._calculate_temporal_entropy_rate(temporal_data)
        
        return results
    
    def _simulate_network_response(self, base_state: np.ndarray, perturbed_state: np.ndarray) -> np.ndarray:
        """Simulate neural network response to perturbation."""
        # Simplified response simulation
        # In practice, this would involve actual network forward pass
        
        # Simple nonlinear transformation
        response = np.tanh(perturbed_state - base_state)
        
        # Add some network-like dynamics
        if len(response) > 1:
            # Simple lateral interactions
            lateral_influence = np.convolve(response, [0.1, 0.8, 0.1], mode='same')
            response = 0.7 * response + 0.3 * lateral_influence
        
        return response
    
    def _calculate_pci_from_responses(self, responses: List[np.ndarray]) -> float:
        """Calculate PCI from perturbation responses."""
        if not responses:
            return 0.0
        
        # Combine all responses
        all_responses = np.array([resp.flatten() for resp in responses])
        
        if all_responses.size == 0:
            return 0.0
        
        # Calculate complexity across response patterns
        complexity_scores = []
        
        # For each response dimension
        for dim in range(all_responses.shape[1]):
            response_series = all_responses[:, dim]
            
            # Binarize responses
            threshold = np.median(response_series)
            binary_series = (response_series > threshold).astype(int)
            
            # Calculate Lempel-Ziv complexity
            lz_complexity = self._lz_complexity_binary(binary_series)
            complexity_scores.append(lz_complexity)
        
        return np.mean(complexity_scores) if complexity_scores else 0.0
        
    def _calculate_parameter_entropy(self, parameters: np.ndarray) -> float:
        """Calculate entropy of parameter distribution."""
        if parameters.size == 0:
            return 0.0
        
        # Discretize parameters for entropy calculation
        n_bins = min(100, max(10, int(np.sqrt(len(parameters)))))
        
        try:
            hist, _ = np.histogram(parameters, bins=n_bins, density=True)
            # Add small epsilon to avoid log(0)
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            
            param_entropy = entropy(hist, base=2)
            return param_entropy
        except:
            return 0.0
    
    def _calculate_parameter_sparsity(self, parameters: np.ndarray) -> float:
        """Calculate sparsity of parameters."""
        if parameters.size == 0:
            return 0.0
        
        # Sparsity as fraction of near-zero parameters
        threshold = np.std(parameters) * 0.1  # 10% of standard deviation
        near_zero = np.abs(parameters) < threshold
        sparsity = np.mean(near_zero)
        
        return float(sparsity)
    
    def _calculate_scale_complexity(self, parameters: np.ndarray) -> float:
        """Calculate complexity across parameter scales."""
        if parameters.size == 0:
            return 0.0
        
        # Calculate complexity at different scales
        scales = [1, 2, 4, 8]
        scale_complexities = []
        
        for scale in scales:
            if scale >= len(parameters):
                continue
            
            # Coarse-grain parameters
            n_groups = len(parameters) // scale
            coarse_params = []
            
            for i in range(n_groups):
                start_idx = i * scale
                end_idx = min(start_idx + scale, len(parameters))
                group_mean = np.mean(parameters[start_idx:end_idx])
                coarse_params.append(group_mean)
            
            if coarse_params:
                coarse_params = np.array(coarse_params)
                scale_entropy = self._calculate_parameter_entropy(coarse_params)
                scale_complexities.append(scale_entropy)
        
        return np.mean(scale_complexities) if scale_complexities else 0.0
    
    def _calculate_layer_complexity_variance(self, model_parameters: Dict[str, np.ndarray]) -> float:
        """Calculate variance in complexity across layers."""
        layer_complexities = []
        
        for param_name, param_values in model_parameters.items():
            if isinstance(param_values, np.ndarray) and param_values.size > 0:
                layer_entropy = self._calculate_parameter_entropy(param_values.flatten())
                layer_complexities.append(layer_entropy)
        
        if len(layer_complexities) > 1:
            return np.var(layer_complexities)
        else:
            return 0.0
    
    def _calculate_lempel_ziv_complexity(self, activations: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity of activations."""
        if activations.size == 0:
            return 0.0
        
        # Calculate LZ complexity for each unit
        lz_scores = []
        
        for unit_idx in range(min(10, activations.shape[0])):  # Limit for efficiency
            unit_activations = activations[unit_idx]
            
            # Binarize activations
            threshold = np.median(unit_activations)
            binary_activations = (unit_activations > threshold).astype(int)
            
            # Calculate LZ complexity
            lz_score = self._lz_complexity_binary(binary_activations)
            lz_scores.append(lz_score)
        
        return np.mean(lz_scores) if lz_scores else 0.0
    
    def _calculate_spectral_complexity(self, activations: np.ndarray) -> float:
        """Calculate spectral complexity of activations."""
        if activations.size == 0:
            return 0.0
        
        spectral_scores = []
        
        for unit_idx in range(min(10, activations.shape[0])):  # Limit for efficiency
            unit_activations = activations[unit_idx]
            
            if len(unit_activations) < 8:  # Need minimum length for FFT
                continue
            
            try:
                # Calculate power spectral density
                freqs, psd = welch(unit_activations, nperseg=min(len(unit_activations), 32))
                
                # Spectral entropy
                psd_norm = psd / np.sum(psd + 1e-10)
                spectral_entropy = entropy(psd_norm + 1e-10, base=2)
                spectral_scores.append(spectral_entropy)
            except:
                continue
        
        return np.mean(spectral_scores) if spectral_scores else 0.0
    
    def _calculate_correlation_complexity(self, activations: np.ndarray) -> float:
        """Calculate correlation-based complexity."""
        if activations.shape[0] < 2:
            return 0.0
        
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(activations)
            
            # Remove diagonal
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            correlations = corr_matrix[mask]
            
            # Complexity as entropy of correlation distribution
            corr_hist, _ = np.histogram(correlations, bins=20, density=True)
            corr_hist = corr_hist + 1e-10
            corr_hist = corr_hist / np.sum(corr_hist)
            
            correlation_entropy = entropy(corr_hist, base=2)
            return correlation_entropy
        except:
            return 0.0
    
    def _calculate_shannon_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        if data.size == 0:
            return 0.0
        
        # Discretize continuous data
        n_bins = min(100, max(10, int(np.sqrt(len(data)))))
        
        try:
            hist, _ = np.histogram(data, bins=n_bins, density=True)
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            
            return entropy(hist, base=2)
        except:
            return 0.0
    
    def _approximate_kolmogorov_complexity(self, data: np.ndarray) -> float:
        """Approximate Kolmogorov complexity using compression."""
        if data.size == 0:
            return 0.0
        
        # Use Lempel-Ziv as Kolmogorov complexity approximation
        # First, convert to binary representation
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
        binary_data = (normalized_data > 0.5).astype(int)
        
        lz_complexity = self._lz_complexity_binary(binary_data)
        
        # Normalize by data length
        normalized_complexity = lz_complexity / len(data) if len(data) > 0 else 0.0
        
        return normalized_complexity
    
    def _approximate_logical_depth(self, data: np.ndarray) -> float:
        """Approximate logical depth."""
        if data.size == 0:
            return 0.0
        
        # Logical depth approximated as number of computational steps
        # needed to generate the pattern from simple rules
        
        # Simple approximation: autocorrelation decay
        if len(data) < 2:
            return 0.0
        
        try:
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find where autocorrelation drops below threshold
            threshold = 0.1 * np.max(autocorr)
            decay_length = np.argmax(autocorr < threshold)
            
            # Normalize by data length
            logical_depth = decay_length / len(data) if len(data) > 0 else 0.0
            
            return min(1.0, logical_depth)
        except:
            return 0.0
    
    def _calculate_effective_complexity(self, data: np.ndarray) -> float:
        """Calculate effective complexity (balance of randomness and regularity)."""
        if data.size == 0:
            return 0.0
        
        # Effective complexity as product of entropy and organization
        shannon_ent = self._calculate_shannon_entropy(data)
        
        # Organization as inverse of randomness
        randomness = self._approximate_kolmogorov_complexity(data)
        organization = 1.0 - randomness
        
        # Effective complexity
        effective_complexity = shannon_ent * organization
        
        return effective_complexity
    
    def _calculate_temporal_lz_complexity(self, temporal_data: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity of temporal patterns."""
        if temporal_data.size == 0:
            return 0.0
        
        # Average across spatial dimensions
        if len(temporal_data.shape) > 1:
            temporal_pattern = np.mean(temporal_data, axis=0)
        else:
            temporal_pattern = temporal_data
        
        # Binarize temporal pattern
        threshold = np.median(temporal_pattern)
        binary_pattern = (temporal_pattern > threshold).astype(int)
        
        return self._lz_complexity_binary(binary_pattern)
    
    def _calculate_temporal_correlation_dimension(self, temporal_data: np.ndarray) -> float:
        """Calculate correlation dimension of temporal data."""
        if temporal_data.size == 0 or len(temporal_data.shape) < 2:
            return 0.0
        
        # Simplified correlation dimension calculation
        # Use first few dimensions for efficiency
        n_dims = min(3, temporal_data.shape[0])
        data_subset = temporal_data[:n_dims]
        
        # Calculate pairwise distances
        n_points = min(100, data_subset.shape[1])  # Limit for efficiency
        
        distances = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                point_i = data_subset[:, i]
                point_j = data_subset[:, j]
                distance = np.linalg.norm(point_i - point_j)
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        # Correlation dimension approximation
        distances = np.array(distances)
        log_distances = np.log(distances + 1e-10)
        
        # Use range of distances to estimate dimension
        distance_range = np.max(log_distances) - np.min(log_distances)
        correlation_dim = distance_range / np.log(len(distances) + 1)
        
        return min(10.0, max(0.0, correlation_dim))  # Reasonable bounds
    
    def _calculate_temporal_predictability(self, temporal_data: np.ndarray) -> float:
        """Calculate predictability of temporal patterns."""
        if temporal_data.size == 0 or temporal_data.shape[1] < 3:
            return 0.0
        
        # Simple predictability: autocorrelation at lag 1
        predictability_scores = []
        
        for dim in range(min(5, temporal_data.shape[0])):  # Limit for efficiency
            time_series = temporal_data[dim]
            
            if len(time_series) < 2:
                continue
            
            # Calculate lag-1 autocorrelation
            try:
                autocorr = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]
                if not np.isnan(autocorr):
                    predictability_scores.append(abs(autocorr))
            except:
                continue
        
        return np.mean(predictability_scores) if predictability_scores else 0.0
    
    def _calculate_temporal_entropy_rate(self, temporal_data: np.ndarray) -> float:
        """Calculate temporal entropy rate."""
        if temporal_data.size == 0:
            return 0.0
        
        # Entropy rate as conditional entropy
        # H(X_t | X_{t-1}, X_{t-2}, ...)
        
        # Simplified: use first-order approximation
        if len(temporal_data.shape) > 1:
            # Average across spatial dimensions
            temporal_pattern = np.mean(temporal_data, axis=0)
        else:
            temporal_pattern = temporal_data
        
        if len(temporal_pattern) < 2:
            return 0.0
        
        # Discretize temporal pattern
        n_bins = min(10, len(temporal_pattern) // 2)
        
        try:
            # Current states
            current_states = temporal_pattern[1:]
            # Previous states  
            previous_states = temporal_pattern[:-1]
            
            # Joint distribution
            joint_hist, _, _ = np.histogram2d(previous_states, current_states, bins=n_bins)
            joint_hist = joint_hist + 1e-10
            joint_hist = joint_hist / np.sum(joint_hist)
            
            # Marginal distribution
            marginal_hist = np.sum(joint_hist, axis=1)
            
            # Conditional entropy H(X_t | X_{t-1})
            conditional_entropy = 0.0
            for i in range(n_bins):
                if marginal_hist[i] > 1e-10:
                    conditional_dist = joint_hist[i] / marginal_hist[i]
                    conditional_dist = conditional_dist + 1e-10
                    conditional_dist = conditional_dist / np.sum(conditional_dist)
                    
                    cond_ent = entropy(conditional_dist, base=2)
                    conditional_entropy += marginal_hist[i] * cond_ent
            
            return conditional_entropy
        except:
            return 0.0
    
    def _lz_complexity_binary(self, binary_sequence: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity of binary sequence."""
        n = len(binary_sequence)
        if n == 0:
            return 0.0
        
        complexity = 0
        i = 0
        
        while i < n:
            k = 1
            while i + k <= n:
                substring = binary_sequence[i:i+k]
                
                # Check if substring exists in previous part
                found = False
                for j in range(i):
                    if j + k <= len(binary_sequence):
                        if np.array_equal(substring, binary_sequence[j:j+k]):
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