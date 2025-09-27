"""
Perturbational Complexity Index (PCI) calculator.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import hilbert
from scipy.stats import zscore


class PCICalculator:
    """
    PCI calculator for consciousness assessment.
    
    Calculates PCI from:
    - TMS-EEG data (gold standard)
    - PCI proxy from spontaneous activity
    - Spatiotemporal complexity measures
    """
    
    def __init__(self, perturbation_intensity: float = 100.0):
        self.perturbation_intensity = perturbation_intensity
    
    def calculate_pci_tms(self, pre_stimulus: np.ndarray, post_stimulus: np.ndarray) -> float:
        """
        Calculate PCI from TMS-EEG data.
        
        Args:
            pre_stimulus: Pre-stimulus EEG data (channels x time)
            post_stimulus: Post-stimulus EEG data (channels x time)
            
        Returns:
            PCI score
        """
        if pre_stimulus.size == 0 or post_stimulus.size == 0:
            return 0.0
        
        # Calculate perturbation response
        response = post_stimulus - pre_stimulus
        
        # Apply source space transformation (simplified)
        source_response = self._transform_to_source_space(response)
        
        # Calculate spatiotemporal complexity
        pci_score = self._calculate_spatiotemporal_complexity(source_response)
        
        # Normalize by perturbation strength
        normalized_pci = pci_score / (self.perturbation_intensity / 100.0)
        
        return float(normalized_pci)
    
    def calculate_pci_proxy(self, eeg_data: np.ndarray) -> float:
        """
        Calculate PCI proxy from spontaneous EEG.
        
        Args:
            eeg_data: Spontaneous EEG data (channels x time)
            
        Returns:
            PCI proxy score
        """
        if eeg_data.size == 0:
            return 0.0
        
        # Use natural fluctuations as "perturbations"
        # Find high-amplitude events
        global_field_power = np.std(eeg_data, axis=0)
        threshold = np.percentile(global_field_power, 90)  # Top 10% events
        
        event_indices = np.where(global_field_power > threshold)[0]
        
        if len(event_indices) < 10:
            return 0.0
        
        # Analyze responses to natural events
        pci_proxy_scores = []
        
        for event_idx in event_indices[:20]:  # Limit for efficiency
            # Define pre and post windows
            pre_window = 50  # 50 samples before
            post_window = 100  # 100 samples after
            
            if event_idx >= pre_window and event_idx + post_window < eeg_data.shape[1]:
                pre_data = eeg_data[:, event_idx - pre_window:event_idx]
                post_data = eeg_data[:, event_idx:event_idx + post_window]
                
                # Calculate response complexity
                response = post_data - np.mean(pre_data, axis=1, keepdims=True)
                complexity = self._calculate_spatiotemporal_complexity(response)
                pci_proxy_scores.append(complexity)
        
        return np.mean(pci_proxy_scores) if pci_proxy_scores else 0.0
    
    def calculate_spatiotemporal_complexity(self, response_data: np.ndarray) -> float:
        """
        Calculate spatiotemporal complexity of neural response.
        
        Args:
            response_data: Neural response data (channels x time)
            
        Returns:
            Spatiotemporal complexity score
        """
        if response_data.size == 0:
            return 0.0
        
        # Spatial complexity
        spatial_complexity = self._calculate_spatial_complexity(response_data)
        
        # Temporal complexity  
        temporal_complexity = self._calculate_temporal_complexity(response_data)
        
        # Combined spatiotemporal complexity
        st_complexity = np.sqrt(spatial_complexity * temporal_complexity)
        
        return float(st_complexity)
    
    def calculate_lempel_ziv_complexity(self, eeg_data: np.ndarray) -> float:
        """
        Calculate Lempel-Ziv complexity from EEG data.
        
        Args:
            eeg_data: EEG data (channels x time)
            
        Returns:
            Average LZ complexity across channels
        """
        if eeg_data.size == 0:
            return 0.0
        
        lz_scores = []
        
        for ch in range(min(eeg_data.shape[0], 10)):  # Limit for efficiency
            channel_data = eeg_data[ch]
            
            # Binarize signal
            threshold = np.median(channel_data)
            binary_signal = (channel_data > threshold).astype(int)
            
            # Calculate LZ complexity
            lz_score = self._lz_complexity_binary(binary_signal)
            lz_scores.append(lz_score)
        
        return np.mean(lz_scores) if lz_scores else 0.0
    
    def _transform_to_source_space(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Transform sensor data to source space (simplified).
        
        In practice, this would use actual head models and inverse solutions.
        Here we use a simplified transformation.
        """
        # Simplified source transformation using spatial filtering
        n_channels, n_timepoints = sensor_data.shape
        
        # Create simplified source montage (bipolar references)
        n_sources = min(n_channels // 2, 10)  # Limit number of sources
        source_data = np.zeros((n_sources, n_timepoints))
        
        for source_idx in range(n_sources):
            # Use pairs of channels as bipolar sources
            ch1_idx = source_idx * 2
            ch2_idx = min(ch1_idx + 1, n_channels - 1)
            
            source_data[source_idx] = sensor_data[ch1_idx] - sensor_data[ch2_idx]
        
        return source_data
    
    def _calculate_spatial_complexity(self, response_data: np.ndarray) -> float:
        """Calculate spatial complexity of response."""
        if response_data.size == 0:
            return 0.0
        
        n_channels, n_timepoints = response_data.shape
        
        # Calculate spatial patterns at each time point
        spatial_patterns = []
        
        for t in range(0, n_timepoints, max(1, n_timepoints // 20)):  # Sample time points
            spatial_pattern = response_data[:, t]
            
            # Normalize spatial pattern
            if np.std(spatial_pattern) > 0:
                spatial_pattern = zscore(spatial_pattern)
                spatial_patterns.append(spatial_pattern)
        
        if len(spatial_patterns) < 2:
            return 0.0
        
        spatial_patterns = np.array(spatial_patterns)
        
        # Calculate diversity of spatial patterns
        pattern_similarities = []
        
        for i in range(len(spatial_patterns)):
            for j in range(i + 1, len(spatial_patterns)):
                corr = np.corrcoef(spatial_patterns[i], spatial_patterns[j])[0, 1]
                if not np.isnan(corr):
                    pattern_similarities.append(abs(corr))
        
        if pattern_similarities:
            # Spatial complexity as inverse of mean similarity
            mean_similarity = np.mean(pattern_similarities)
            spatial_complexity = 1.0 - mean_similarity
        else:
            spatial_complexity = 0.0
        
        return max(0.0, spatial_complexity)
    
    def _calculate_temporal_complexity(self, response_data: np.ndarray) -> float:
        """Calculate temporal complexity of response."""
        if response_data.size == 0:
            return 0.0
        
        # Calculate temporal complexity using Lempel-Ziv on time series
        temporal_complexities = []
        
        for ch in range(min(response_data.shape[0], 10)):  # Limit for efficiency
            channel_response = response_data[ch]
            
            # Binarize temporal response
            threshold = np.median(channel_response)
            binary_response = (channel_response > threshold).astype(int)
            
            # Calculate temporal LZ complexity
            temporal_lz = self._lz_complexity_binary(binary_response)
            temporal_complexities.append(temporal_lz)
        
        return np.mean(temporal_complexities) if temporal_complexities else 0.0
    
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
    
    def calculate_perturbational_response_metrics(self, pre_data: np.ndarray, post_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive perturbational response metrics.
        
        Args:
            pre_data: Pre-perturbation data
            post_data: Post-perturbation data
            
        Returns:
            Dictionary with response metrics
        """
        results = {}
        
        if pre_data.size == 0 or post_data.size == 0:
            return results
        
        # Response amplitude
        response = post_data - pre_data
        results['response_amplitude'] = np.mean(np.abs(response))
        results['response_peak'] = np.max(np.abs(response))
        
        # Response duration
        response_envelope = np.sqrt(np.mean(response**2, axis=0))  # RMS across channels
        
        # Find significant response period
        threshold = np.max(response_envelope) * 0.1  # 10% of peak
        significant_indices = response_envelope > threshold
        
        if np.any(significant_indices):
            # Duration of significant response
            response_duration = np.sum(significant_indices)
            results['response_duration'] = response_duration
            
            # Response latency (time to peak)
            peak_idx = np.argmax(response_envelope)
            results['response_latency'] = peak_idx
        else:
            results['response_duration'] = 0
            results['response_latency'] = 0
        
        # Response complexity
        results['response_complexity'] = self.calculate_spatiotemporal_complexity(response)
        
        # Response propagation (how response spreads across channels)
        if response.shape[0] > 1:
            # Calculate cross-channel correlations in response
            response_correlations = []
            for i in range(response.shape[0]):
                for j in range(i + 1, response.shape[0]):
                    corr = np.corrcoef(response[i], response[j])[0, 1]
                    if not np.isnan(corr):
                        response_correlations.append(abs(corr))
            
            if response_correlations:
                results['response_coherence'] = np.mean(response_correlations)
                results['response_propagation'] = 1.0 - np.var(response_correlations)  # Lower variance = better propagation
            else:
                results['response_coherence'] = 0.0
                results['response_propagation'] = 0.0
        
        return results