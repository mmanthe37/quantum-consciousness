"""
Quantum coherence calculator for consciousness research.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple


class QuantumCoherenceCalculator:
    """
    Calculator for quantum coherence measures in consciousness research.
    """
    
    def __init__(self, coherence_threshold: float = 0.7):
        self.coherence_threshold = coherence_threshold
    
    def calculate_state_coherence(self, quantum_state: Union[complex, np.ndarray]) -> Dict[str, float]:
        """Calculate coherence properties of quantum state."""
        results = {}
        
        if isinstance(quantum_state, complex):
            # Single qubit state
            coherence = abs(quantum_state)
            results['quantum_coherence'] = float(coherence)
            results['coherence_threshold_exceeded'] = float(coherence > self.coherence_threshold)
        
        elif isinstance(quantum_state, np.ndarray):
            # Multi-qubit or continuous state
            if quantum_state.dtype == complex:
                # Complex quantum state
                coherence_measure = self._calculate_quantum_coherence_complex(quantum_state)
                results['quantum_coherence'] = coherence_measure
                results['coherence_threshold_exceeded'] = float(coherence_measure > self.coherence_threshold)
            else:
                # Real-valued approximation
                coherence_measure = self._calculate_coherence_approximation(quantum_state)
                results['quantum_coherence'] = coherence_measure
                results['coherence_threshold_exceeded'] = float(coherence_measure > self.coherence_threshold)
        
        return results
    
    def calculate_neural_quantum_coherence(self, neural_data: np.ndarray) -> Dict[str, float]:
        """Calculate quantum coherence from neural data."""
        results = {}
        
        if neural_data.size == 0:
            return results
        
        # Quantum information integration
        results['quantum_information_integration'] = self._calculate_quantum_information_integration(neural_data)
        
        # Neural quantum coherence
        results['neural_quantum_coherence'] = self._calculate_neural_quantum_coherence(neural_data)
        
        # Quantum entanglement proxy
        results['quantum_entanglement_proxy'] = self._calculate_entanglement_proxy(neural_data)
        
        return results
    
    def _calculate_quantum_coherence_complex(self, quantum_state: np.ndarray) -> float:
        """Calculate coherence for complex quantum state."""
        if quantum_state.size == 0:
            return 0.0
        
        # For density matrix representation
        if len(quantum_state.shape) == 2 and quantum_state.shape[0] == quantum_state.shape[1]:
            # Quantum coherence via l1-norm of off-diagonal elements
            off_diagonal = quantum_state - np.diag(np.diag(quantum_state))
            coherence = np.sum(np.abs(off_diagonal))
            return float(coherence)
        
        # For state vector representation
        else:
            # Coherence as deviation from classical mixed state
            probabilities = np.abs(quantum_state)**2
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
                
                # Von Neumann entropy (lower = more coherent)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                max_entropy = np.log2(len(probabilities))
                
                # Coherence as 1 - normalized entropy
                coherence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
                return float(coherence)
            else:
                return 0.0
    
    def _calculate_coherence_approximation(self, data: np.ndarray) -> float:
        """Calculate coherence approximation from real-valued data."""
        if data.size == 0:
            return 0.0
        
        # Use correlation structure as coherence proxy
        if len(data.shape) > 1:
            # Multi-dimensional data
            correlation_matrix = np.corrcoef(data)
            
            # Coherence as mean off-diagonal correlation strength
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            coherence = np.mean(np.abs(correlation_matrix[mask]))
            
            return float(coherence)
        else:
            # Single dimension - use autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                
                # Coherence as integrated autocorrelation
                coherence = np.sum(np.abs(autocorr)) / len(autocorr)
                return float(coherence)
            else:
                return 0.0
    
    def _calculate_quantum_information_integration(self, neural_data: np.ndarray) -> float:
        """Calculate quantum information integration measure."""
        if neural_data.size == 0:
            return 0.0
        
        # Simplified quantum information integration
        # Based on mutual information between neural regions
        
        if len(neural_data.shape) < 2:
            neural_data = neural_data.reshape(1, -1)
        
        n_regions = neural_data.shape[0]
        
        if n_regions < 2:
            return 0.0
        
        # Calculate mutual information matrix
        mutual_info_matrix = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(i, n_regions):
                if i == j:
                    mutual_info_matrix[i, j] = 1.0
                else:
                    mi = self._calculate_mutual_information(neural_data[i], neural_data[j])
                    mutual_info_matrix[i, j] = mi
                    mutual_info_matrix[j, i] = mi
        
        # Quantum information integration as mean mutual information
        upper_triangle = mutual_info_matrix[np.triu_indices_from(mutual_info_matrix, k=1)]
        integration_score = np.mean(upper_triangle)
        
        return float(integration_score)
    
    def _calculate_neural_quantum_coherence(self, neural_data: np.ndarray) -> float:
        """Calculate neural quantum coherence."""
        if neural_data.size == 0:
            return 0.0
        
        # Use phase coherence as quantum coherence proxy
        from scipy.signal import hilbert
        
        if len(neural_data.shape) < 2:
            neural_data = neural_data.reshape(1, -1)
        
        coherence_values = []
        
        # Calculate pairwise phase coherence
        for i in range(min(neural_data.shape[0], 10)):  # Limit for efficiency
            for j in range(i + 1, min(neural_data.shape[0], 10)):
                try:
                    # Extract analytic signals
                    analytic_i = hilbert(neural_data[i])
                    analytic_j = hilbert(neural_data[j])
                    
                    # Phase difference
                    phase_i = np.angle(analytic_i)
                    phase_j = np.angle(analytic_j)
                    phase_diff = phase_i - phase_j
                    
                    # Phase locking value (coherence measure)
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    coherence_values.append(plv)
                except:
                    continue
        
        return np.mean(coherence_values) if coherence_values else 0.0
    
    def _calculate_entanglement_proxy(self, neural_data: np.ndarray) -> float:
        """Calculate quantum entanglement proxy from neural data."""
        if neural_data.size == 0:
            return 0.0
        
        # Entanglement proxy using non-local correlations
        if len(neural_data.shape) < 2:
            return 0.0
        
        n_regions = neural_data.shape[0]
        
        if n_regions < 2:
            return 0.0
        
        # Calculate delayed correlations (non-local connections)
        delays = [1, 2, 3, 5, 8]  # Sample delays
        non_local_correlations = []
        
        for delay in delays:
            if delay >= neural_data.shape[1]:
                continue
            
            for i in range(min(n_regions, 10)):  # Limit for efficiency
                for j in range(i + 1, min(n_regions, 10)):
                    # Delayed correlation
                    try:
                        signal_i = neural_data[i, :-delay]
                        signal_j = neural_data[j, delay:]
                        
                        if len(signal_i) > 10:  # Need sufficient data
                            corr = np.corrcoef(signal_i, signal_j)[0, 1]
                            if not np.isnan(corr):
                                non_local_correlations.append(abs(corr))
                    except:
                        continue
        
        # Entanglement proxy as mean non-local correlation
        entanglement_proxy = np.mean(non_local_correlations) if non_local_correlations else 0.0
        
        return float(entanglement_proxy)
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Calculate mutual information between two signals."""
        try:
            # Discretize signals
            x_discrete = np.digitize(x, np.histogram(x, bins=bins)[1][:-1])
            y_discrete = np.digitize(y, np.histogram(y, bins=bins)[1][:-1])
            
            # Joint histogram
            joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=bins)
            joint_hist = joint_hist + 1e-10  # Avoid log(0)
            joint_prob = joint_hist / np.sum(joint_hist)
            
            # Marginal histograms
            x_hist = np.sum(joint_prob, axis=1)
            y_hist = np.sum(joint_prob, axis=0)
            
            # Mutual information
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if joint_prob[i, j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_hist[i] * y_hist[j]))
            
            return mi
        except:
            return 0.0