"""
EEG analysis for consciousness assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import signal
from scipy.stats import entropy
from scipy.signal import hilbert, welch


class EEGAnalyzer:
    """
    EEG analyzer for consciousness indicators.
    
    Analyzes EEG signals for:
    - Spectral features and power bands
    - Complexity measures
    - Microstates
    - Global connectivity measures
    """
    
    def __init__(self, sampling_rate: int = 250, filter_low: float = 0.5, filter_high: float = 45.0):
        self.sampling_rate = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
    
    def analyze_spectral_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze spectral features of EEG data.
        
        Args:
            eeg_data: EEG data (channels x time)
            
        Returns:
            Dictionary with spectral features
        """
        results = {}
        
        if eeg_data.size == 0:
            return results
        
        # Filter EEG data
        filtered_eeg = self._filter_eeg(eeg_data)
        
        # Calculate power spectral density
        freqs, psd = self._calculate_psd(filtered_eeg)
        
        # Power band analysis
        band_powers = self._calculate_band_powers(freqs, psd)
        results.update(band_powers)
        
        # Spectral entropy
        results['spectral_entropy'] = self._calculate_spectral_entropy(psd)
        
        # Peak frequency
        results['peak_frequency'] = self._calculate_peak_frequency(freqs, psd)
        
        # Spectral edge frequency
        results['spectral_edge_frequency'] = self._calculate_spectral_edge_frequency(freqs, psd)
        
        # Alpha/delta ratio (consciousness indicator)
        if 'alpha_power' in results and 'delta_power' in results:
            results['alpha_delta_ratio'] = results['alpha_power'] / (results['delta_power'] + 1e-10)
        
        return results
    
    def calculate_complexity_measures(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate complexity measures from EEG data.
        
        Args:
            eeg_data: EEG data (channels x time)
            
        Returns:
            Dictionary with complexity measures
        """
        results = {}
        
        if eeg_data.size == 0:
            return results
        
        # Filter EEG data
        filtered_eeg = self._filter_eeg(eeg_data)
        
        # Lempel-Ziv complexity
        results['lz_complexity'] = self._calculate_lz_complexity(filtered_eeg)
        
        # Sample entropy
        results['sample_entropy'] = self._calculate_sample_entropy(filtered_eeg)
        
        # Permutation entropy  
        results['permutation_entropy'] = self._calculate_permutation_entropy(filtered_eeg)
        
        # Multiscale entropy
        results['multiscale_entropy'] = self._calculate_multiscale_entropy(filtered_eeg)
        
        # Higuchi fractal dimension
        results['fractal_dimension'] = self._calculate_fractal_dimension(filtered_eeg)
        
        return results
    
    def analyze_microstates(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze EEG microstates.
        
        Args:
            eeg_data: EEG data (channels x time)
            
        Returns:
            Dictionary with microstate metrics
        """
        results = {}
        
        if eeg_data.size == 0:
            return results
        
        # Filter EEG data
        filtered_eeg = self._filter_eeg(eeg_data)
        
        # Global field power
        gfp = self._calculate_global_field_power(filtered_eeg)
        results['mean_gfp'] = np.mean(gfp)
        results['gfp_variance'] = np.var(gfp)
        
        # Microstate segmentation (simplified)
        microstate_metrics = self._analyze_microstate_dynamics(filtered_eeg, gfp)
        results.update(microstate_metrics)
        
        return results
    
    def calculate_global_measures(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate global EEG measures.
        
        Args:
            eeg_data: EEG data (channels x time)
            
        Returns:
            Dictionary with global measures
        """
        results = {}
        
        if eeg_data.size == 0:
            return results
        
        # Filter EEG data
        filtered_eeg = self._filter_eeg(eeg_data)
        
        # Global synchronization
        results['global_synchronization'] = self._calculate_global_synchronization(filtered_eeg)
        
        # Phase lag index
        results['phase_lag_index'] = self._calculate_phase_lag_index(filtered_eeg)
        
        # Omega complexity
        results['omega_complexity'] = self._calculate_omega_complexity(filtered_eeg)
        
        # Signal to noise ratio
        results['signal_to_noise_ratio'] = self._calculate_snr(filtered_eeg)
        
        return results
    
    def _filter_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to EEG data."""
        if eeg_data.size == 0:
            return eeg_data
        
        # Design Butterworth bandpass filter
        nyquist = self.sampling_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(eeg_data)
            for ch in range(eeg_data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, eeg_data[ch])
            
            return filtered_data
        except:
            # Return original data if filtering fails
            return eeg_data
    
    def _calculate_psd(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density."""
        try:
            # Average PSD across channels
            channel_psds = []
            for ch in range(eeg_data.shape[0]):
                freqs, psd = welch(eeg_data[ch], self.sampling_rate, nperseg=min(256, eeg_data.shape[1]//4))
                channel_psds.append(psd)
            
            avg_psd = np.mean(channel_psds, axis=0)
            return freqs, avg_psd
        except:
            # Fallback
            freqs = np.linspace(0, self.sampling_rate/2, 100)
            psd = np.ones(100)
            return freqs, psd
    
    def _calculate_band_powers(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Calculate power in different frequency bands."""
        bands = {
            'delta_power': (0.5, 4),
            'theta_power': (4, 8),
            'alpha_power': (8, 13),
            'beta_power': (13, 30),
            'gamma_power': (30, 45)
        }
        
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[band_name] = float(band_power)
            else:
                band_powers[band_name] = 0.0
        
        # Total power
        total_power = np.trapz(psd, freqs)
        band_powers['total_power'] = float(total_power)
        
        # Relative powers
        for band_name in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']:
            if total_power > 0:
                relative_name = band_name.replace('_power', '_relative_power')
                band_powers[relative_name] = band_powers[band_name] / total_power
            else:
                band_powers[band_name.replace('_power', '_relative_power')] = 0.0
        
        return band_powers
    
    def _calculate_spectral_entropy(self, psd: np.ndarray) -> float:
        """Calculate spectral entropy."""
        if psd.size == 0 or np.sum(psd) == 0:
            return 0.0
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy
        spectral_entropy = entropy(psd_norm + 1e-10, base=2)
        
        return float(spectral_entropy)
    
    def _calculate_peak_frequency(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Calculate peak frequency."""
        if psd.size == 0:
            return 0.0
        
        peak_idx = np.argmax(psd)
        return float(freqs[peak_idx])
    
    def _calculate_spectral_edge_frequency(self, freqs: np.ndarray, psd: np.ndarray, percentile: float = 95) -> float:
        """Calculate spectral edge frequency."""
        if psd.size == 0:
            return 0.0
        
        # Cumulative power
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        
        if total_power == 0:
            return 0.0
        
        # Find frequency at percentile
        threshold = (percentile / 100) * total_power
        edge_idx = np.argmax(cumulative_power >= threshold)
        
        return float(freqs[edge_idx])
    
    def _calculate_lz_complexity(self, eeg_data: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity."""
        if eeg_data.size == 0:
            return 0.0
        
        # Average across channels
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
    
    def _calculate_sample_entropy(self, eeg_data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        if eeg_data.size == 0:
            return 0.0
        
        # Average across channels
        entropy_scores = []
        
        for ch in range(min(eeg_data.shape[0], 10)):  # Limit for efficiency
            channel_data = eeg_data[ch]
            
            if len(channel_data) < m + 1:
                continue
            
            # Normalize data
            data_std = np.std(channel_data)
            if data_std == 0:
                continue
            
            normalized_data = (channel_data - np.mean(channel_data)) / data_std
            tolerance = r * data_std
            
            # Calculate sample entropy
            sample_ent = self._sample_entropy_calculation(normalized_data, m, tolerance)
            if not np.isnan(sample_ent) and not np.isinf(sample_ent):
                entropy_scores.append(sample_ent)
        
        return np.mean(entropy_scores) if entropy_scores else 0.0
    
    def _calculate_permutation_entropy(self, eeg_data: np.ndarray, order: int = 3) -> float:
        """Calculate permutation entropy."""
        if eeg_data.size == 0:
            return 0.0
        
        # Average across channels
        entropy_scores = []
        
        for ch in range(min(eeg_data.shape[0], 10)):  # Limit for efficiency
            channel_data = eeg_data[ch]
            
            if len(channel_data) < order:
                continue
            
            # Calculate permutation entropy
            perm_ent = self._permutation_entropy_calculation(channel_data, order)
            if not np.isnan(perm_ent):
                entropy_scores.append(perm_ent)
        
        return np.mean(entropy_scores) if entropy_scores else 0.0
    
    def _calculate_multiscale_entropy(self, eeg_data: np.ndarray, max_scale: int = 10) -> float:
        """Calculate multiscale entropy."""
        if eeg_data.size == 0:
            return 0.0
        
        # Average across channels
        mse_scores = []
        
        for ch in range(min(eeg_data.shape[0], 5)):  # Limit for efficiency
            channel_data = eeg_data[ch]
            
            scale_entropies = []
            for scale in range(1, min(max_scale + 1, len(channel_data) // 10)):
                # Coarse-grain the time series
                coarse_grained = self._coarse_grain(channel_data, scale)
                
                if len(coarse_grained) > 10:  # Need sufficient data
                    scale_entropy = self._calculate_sample_entropy(
                        coarse_grained.reshape(1, -1)
                    )
                    if not np.isnan(scale_entropy):
                        scale_entropies.append(scale_entropy)
            
            if scale_entropies:
                mse_scores.append(np.mean(scale_entropies))
        
        return np.mean(mse_scores) if mse_scores else 0.0
    
    def _calculate_fractal_dimension(self, eeg_data: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension."""
        if eeg_data.size == 0:
            return 0.0
        
        # Average across channels
        fd_scores = []
        
        for ch in range(min(eeg_data.shape[0], 10)):  # Limit for efficiency
            channel_data = eeg_data[ch]
            
            if len(channel_data) < 100:  # Need sufficient data
                continue
            
            fd_score = self._higuchi_fractal_dimension(channel_data)
            if not np.isnan(fd_score) and fd_score > 0:
                fd_scores.append(fd_score)
        
        return np.mean(fd_scores) if fd_scores else 1.0
    
    def _calculate_global_field_power(self, eeg_data: np.ndarray) -> np.ndarray:
        """Calculate global field power."""
        if eeg_data.size == 0:
            return np.array([])
        
        # GFP as standard deviation across channels at each time point
        gfp = np.std(eeg_data, axis=0)
        
        return gfp
    
    def _analyze_microstate_dynamics(self, eeg_data: np.ndarray, gfp: np.ndarray) -> Dict[str, float]:
        """Analyze microstate dynamics (simplified)."""
        results = {}
        
        if gfp.size == 0:
            return results
        
        # Find GFP peaks (simplified microstate identification)
        # In practice, would use topographic clustering
        gfp_threshold = np.percentile(gfp, 75)  # Top 25% of GFP values
        peaks = gfp > gfp_threshold
        
        if np.any(peaks):
            # Microstate duration (simplified)
            state_changes = np.diff(peaks.astype(int))
            state_durations = []
            
            in_state = False
            duration = 0
            
            for change in state_changes:
                if change == 1:  # Entering microstate
                    in_state = True
                    duration = 1
                elif change == -1 and in_state:  # Leaving microstate
                    state_durations.append(duration)
                    in_state = False
                elif in_state:
                    duration += 1
            
            if state_durations:
                results['mean_microstate_duration'] = np.mean(state_durations) / self.sampling_rate * 1000  # ms
                results['microstate_duration_std'] = np.std(state_durations) / self.sampling_rate * 1000  # ms
            else:
                results['mean_microstate_duration'] = 0.0
                results['microstate_duration_std'] = 0.0
            
            # Microstate coverage
            results['microstate_coverage'] = np.mean(peaks)
        else:
            results['mean_microstate_duration'] = 0.0
            results['microstate_duration_std'] = 0.0
            results['microstate_coverage'] = 0.0
        
        return results
    
    def _calculate_global_synchronization(self, eeg_data: np.ndarray) -> float:
        """Calculate global synchronization."""
        if eeg_data.shape[0] < 2:
            return 0.0
        
        # Calculate pairwise phase synchronization
        synchronization_values = []
        
        for i in range(min(10, eeg_data.shape[0])):  # Limit for efficiency
            for j in range(i + 1, min(10, eeg_data.shape[0])):
                # Extract instantaneous phases
                analytic_i = hilbert(eeg_data[i])
                analytic_j = hilbert(eeg_data[j])
                
                phase_i = np.angle(analytic_i)
                phase_j = np.angle(analytic_j)
                
                # Phase locking value
                phase_diff = phase_i - phase_j
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                
                synchronization_values.append(plv)
        
        return np.mean(synchronization_values) if synchronization_values else 0.0
    
    def _calculate_phase_lag_index(self, eeg_data: np.ndarray) -> float:
        """Calculate phase lag index."""
        if eeg_data.shape[0] < 2:
            return 0.0
        
        pli_values = []
        
        for i in range(min(10, eeg_data.shape[0])):  # Limit for efficiency
            for j in range(i + 1, min(10, eeg_data.shape[0])):
                # Extract instantaneous phases
                analytic_i = hilbert(eeg_data[i])
                analytic_j = hilbert(eeg_data[j])
                
                phase_i = np.angle(analytic_i)
                phase_j = np.angle(analytic_j)
                
                # Phase lag index
                phase_diff = phase_i - phase_j
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                
                pli_values.append(pli)
        
        return np.mean(pli_values) if pli_values else 0.0
    
    def _calculate_omega_complexity(self, eeg_data: np.ndarray) -> float:
        """Calculate omega complexity."""
        if eeg_data.shape[0] < 2:
            return 0.0
        
        try:
            # Covariance matrix
            cov_matrix = np.cov(eeg_data)
            
            # Eigenvalues
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = eigenvals[eigenvals > 0]  # Remove negative/zero eigenvalues
            
            if len(eigenvals) == 0:
                return 0.0
            
            # Normalize eigenvalues
            eigenvals_norm = eigenvals / np.sum(eigenvals)
            
            # Omega complexity
            omega = -np.sum(eigenvals_norm * np.log(eigenvals_norm + 1e-10))
            
            return float(omega)
        except:
            return 0.0
    
    def _calculate_snr(self, eeg_data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if eeg_data.size == 0:
            return 0.0
        
        # Simple SNR estimation
        signal_power = np.var(eeg_data)
        
        # Estimate noise as high-frequency component
        try:
            # High-pass filter to extract noise
            b, a = signal.butter(4, 20 / (self.sampling_rate / 2), btype='high')
            noise_estimates = []
            
            for ch in range(eeg_data.shape[0]):
                noise_component = signal.filtfilt(b, a, eeg_data[ch])
                noise_power = np.var(noise_component)
                noise_estimates.append(noise_power)
            
            noise_power = np.mean(noise_estimates)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return float(snr)
            else:
                return float('inf')
        except:
            return 10.0  # Default reasonable SNR
    
    # Helper methods for complexity calculations
    
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
    
    def _sample_entropy_calculation(self, data: np.ndarray, m: int, r: float) -> float:
        """Calculate sample entropy."""
        N = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= r:
                        C[i] += 1
            
            phi = np.mean(np.log(C / (N - m + 1.0)))
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return np.nan
    
    def _permutation_entropy_calculation(self, data: np.ndarray, order: int) -> float:
        """Calculate permutation entropy."""
        try:
            # Generate ordinal patterns
            ordinal_patterns = []
            
            for i in range(len(data) - order + 1):
                segment = data[i:i + order]
                sorted_indices = np.argsort(segment)
                ordinal_patterns.append(tuple(sorted_indices))
            
            # Calculate relative frequencies
            from collections import Counter
            pattern_counts = Counter(ordinal_patterns)
            
            total_patterns = len(ordinal_patterns)
            probabilities = [count / total_patterns for count in pattern_counts.values()]
            
            # Calculate entropy
            return entropy(probabilities, base=2)
        except:
            return np.nan
    
    def _coarse_grain(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain time series."""
        N = len(data)
        coarse_grained = []
        
        for i in range(0, N, scale):
            end_idx = min(i + scale, N)
            coarse_grained.append(np.mean(data[i:end_idx]))
        
        return np.array(coarse_grained)
    
    def _higuchi_fractal_dimension(self, data: np.ndarray, kmax: int = 10) -> float:
        """Calculate Higuchi fractal dimension."""
        try:
            N = len(data)
            L = []
            x = []
            
            for k in range(1, kmax + 1):
                Lk = []
                
                for m in range(k):
                    Lmk = 0
                    
                    for i in range(1, int((N - m) / k)):
                        Lmk += abs(data[m + i * k] - data[m + (i - 1) * k])
                    
                    normalization = (N - 1) / (k * int((N - m) / k) * k)
                    Lmk *= normalization
                    Lmk.append(Lmk)
                
                L.append(np.mean(Lmk))
                x.append(np.log(1.0 / k))
            
            # Linear regression
            coeffs = np.polyfit(x, np.log(L), 1)
            return coeffs[0]
        except:
            return np.nan