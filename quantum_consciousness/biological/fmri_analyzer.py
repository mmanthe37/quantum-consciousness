"""
fMRI analysis for consciousness assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.signal import correlate
import networkx as nx


class fMRIAnalyzer:
    """
    fMRI analyzer for consciousness indicators.
    
    Analyzes fMRI signals for:
    - Brain network topology
    - Default mode network activity
    - Global workspace dynamics
    - Criticality measures
    """
    
    def __init__(self, tr: float = 2.0, smoothing_kernel: float = 6.0):
        self.tr = tr  # Repetition time
        self.smoothing_kernel = smoothing_kernel
    
    def analyze_brain_networks(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """Analyze brain network properties."""
        results = {}
        
        if fmri_data.size == 0:
            return results
        
        # Calculate connectivity matrix
        connectivity_matrix = np.corrcoef(fmri_data)
        
        # Create network graph
        threshold = 0.3  # Correlation threshold
        binary_matrix = (np.abs(connectivity_matrix) > threshold).astype(int)
        np.fill_diagonal(binary_matrix, 0)
        
        G = nx.from_numpy_array(binary_matrix)
        
        # Network metrics
        if len(G) > 0:
            results['clustering_coefficient'] = nx.average_clustering(G)
            results['network_density'] = nx.density(G)
            
            if nx.is_connected(G):
                results['characteristic_path_length'] = nx.average_shortest_path_length(G)
                results['global_efficiency'] = nx.global_efficiency(G)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len, default=[])
                if len(largest_cc) > 1:
                    subgraph = G.subgraph(largest_cc)
                    results['characteristic_path_length'] = nx.average_shortest_path_length(subgraph)
                    results['global_efficiency'] = nx.global_efficiency(subgraph)
                else:
                    results['characteristic_path_length'] = 0
                    results['global_efficiency'] = 0
            
            # Small-world coefficient
            if results.get('clustering_coefficient', 0) > 0 and results.get('characteristic_path_length', 0) > 0:
                # Random network comparison
                n = len(G)
                m = G.number_of_edges()
                if m > 0:
                    p = 2 * m / (n * (n - 1))
                    C_rand = p
                    L_rand = np.log(n) / np.log(2 * m / n) if 2 * m / n > 1 else 1
                    
                    if C_rand > 0 and L_rand > 0:
                        sigma = (results['clustering_coefficient'] / C_rand) / (results['characteristic_path_length'] / L_rand)
                        results['small_world_coefficient'] = sigma
        
        return results
    
    def analyze_default_mode_network(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """Analyze default mode network activity."""
        results = {}
        
        if fmri_data.size == 0:
            return results
        
        # Simplified DMN analysis - would normally use specific ROIs
        # Here we use posterior regions as DMN proxy
        n_regions = fmri_data.shape[0]
        dmn_regions = list(range(max(1, n_regions // 2), n_regions))  # Posterior half as proxy
        
        if dmn_regions:
            dmn_data = fmri_data[dmn_regions]
            
            # DMN connectivity
            dmn_connectivity = np.corrcoef(dmn_data)
            results['dmn_internal_connectivity'] = np.mean(dmn_connectivity[np.triu_indices_from(dmn_connectivity, k=1)])
            
            # DMN deactivation (negative correlation with task-positive regions)
            task_positive_regions = list(range(0, n_regions // 2))  # Anterior half as proxy
            if task_positive_regions:
                task_positive_data = fmri_data[task_positive_regions]
                
                # Cross-network correlations
                cross_correlations = []
                for dmn_region in range(len(dmn_regions)):
                    for tp_region in range(len(task_positive_regions)):
                        corr = np.corrcoef(dmn_data[dmn_region], task_positive_data[tp_region])[0, 1]
                        if not np.isnan(corr):
                            cross_correlations.append(corr)
                
                if cross_correlations:
                    results['dmn_task_positive_anticorrelation'] = -np.mean(cross_correlations)  # Negative for anticorrelation
        
        return results
    
    def analyze_global_workspace(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """Analyze global workspace characteristics."""
        results = {}
        
        if fmri_data.size == 0:
            return results
        
        # Information integration across brain regions
        connectivity_matrix = np.corrcoef(fmri_data)
        
        # Global integration as mean connectivity strength
        upper_triangle = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
        results['global_integration'] = np.mean(np.abs(upper_triangle))
        
        # Information broadcasting (variance in connectivity)
        results['information_broadcasting'] = np.var(upper_triangle)
        
        # Participation coefficient (how much each region connects to different modules)
        # Simplified version without community detection
        region_connectivity_variance = []
        for i in range(connectivity_matrix.shape[0]):
            region_connections = connectivity_matrix[i, :]
            region_connections = region_connections[region_connections != 1]  # Remove self-connection
            if len(region_connections) > 0:
                region_connectivity_variance.append(np.var(region_connections))
        
        if region_connectivity_variance:
            results['participation_coefficient'] = np.mean(region_connectivity_variance)
        
        return results
    
    def calculate_criticality_measures(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """Calculate brain criticality measures."""
        results = {}
        
        if fmri_data.size == 0:
            return results
        
        # Avalanche analysis (simplified)
        # In practice, would analyze neuronal avalanches
        
        # Power law analysis of amplitude fluctuations
        for region_idx in range(min(5, fmri_data.shape[0])):  # Limit for efficiency
            region_data = fmri_data[region_idx]
            
            # Calculate power spectrum
            from scipy.signal import welch
            freqs, psd = welch(region_data, nperseg=min(64, len(region_data)//4))
            
            # Fit power law to 1/f spectrum
            if len(freqs) > 5:
                # Log-log regression
                log_freqs = np.log10(freqs[1:])  # Skip DC component
                log_psd = np.log10(psd[1:])
                
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(log_freqs, log_psd)
                    
                    # Store results for first region (as example)
                    if region_idx == 0:
                        results['power_law_exponent'] = -slope  # Negative because of 1/f
                        results['power_law_fit_quality'] = r_value**2
                        
                        # Criticality indicator (exponent close to 1 for pink noise)
                        criticality_score = 1.0 - abs(-slope - 1.0)  # Closer to 1 is more critical
                        results['criticality_index'] = max(0, criticality_score)
                except:
                    if region_idx == 0:
                        results['power_law_exponent'] = 0
                        results['power_law_fit_quality'] = 0
                        results['criticality_index'] = 0
        
        # Long-range temporal correlations (DFA - simplified)
        results['long_range_correlations'] = self._calculate_dfa_exponent(fmri_data)
        
        return results
    
    def _calculate_dfa_exponent(self, fmri_data: np.ndarray) -> float:
        """Calculate detrended fluctuation analysis exponent (simplified)."""
        if fmri_data.size == 0:
            return 0.0
        
        # Use first region as representative
        data = fmri_data[0] if len(fmri_data.shape) > 1 else fmri_data
        
        if len(data) < 50:
            return 0.0
        
        # Integrate the signal
        integrated_data = np.cumsum(data - np.mean(data))
        
        # Define scales
        min_scale = 10
        max_scale = len(data) // 4
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 10).astype(int)
        scales = np.unique(scales)
        
        fluctuations = []
        
        for scale in scales:
            if scale >= len(integrated_data):
                continue
            
            # Divide signal into non-overlapping segments
            n_segments = len(integrated_data) // scale
            
            if n_segments < 2:
                continue
            
            segment_fluctuations = []
            
            for i in range(n_segments):
                start_idx = i * scale
                end_idx = start_idx + scale
                segment = integrated_data[start_idx:end_idx]
                
                # Detrend (linear fit)
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((segment - trend)**2))
                segment_fluctuations.append(fluctuation)
            
            if segment_fluctuations:
                fluctuations.append(np.mean(segment_fluctuations))
        
        if len(fluctuations) < 3:
            return 0.5  # Default neutral value
        
        # Fit power law
        try:
            log_scales = np.log10(scales[:len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)
            
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_fluctuations)
            
            # Return DFA exponent (should be around 1.0 for 1/f noise)
            return float(slope)
        except:
            return 0.5