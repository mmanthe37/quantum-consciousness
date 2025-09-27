"""
Main biological consciousness measurement processor.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.base import BaseProcessor, ProcessingResult, DataValidator
from .eeg_analyzer import EEGAnalyzer
from .fmri_analyzer import fMRIAnalyzer
from .connectivity_analyzer import ConnectivityAnalyzer
from .pci_calculator import PCICalculator


class BiologicalProcessor(BaseProcessor):
    """
    Biological consciousness measurement processor.
    
    Processes biological data including:
    - EEG signals for consciousness assessment
    - fMRI data for network analysis
    - Dynamic functional connectivity
    - Perturbational Complexity Index (PCI) for biological systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.eeg_sampling_rate = self.config.get('eeg_sampling_rate', 250)
        self.eeg_filter_low = self.config.get('eeg_filter_low', 0.5)
        self.eeg_filter_high = self.config.get('eeg_filter_high', 45.0)
        self.fmri_tr = self.config.get('fmri_tr', 2.0)
        self.fmri_smoothing_kernel = self.config.get('fmri_smoothing_kernel', 6.0)
        self.pci_perturbation_intensity = self.config.get('pci_perturbation_intensity', 100.0)
        self.connectivity_method = self.config.get('connectivity_method', 'pearson')
        
        # Sub-processors
        self.eeg_analyzer = None
        self.fmri_analyzer = None
        self.connectivity_analyzer = None
        self.pci_calculator = None
    
    def initialize(self) -> bool:
        """Initialize biological processor components."""
        try:
            self.eeg_analyzer = EEGAnalyzer(
                sampling_rate=self.eeg_sampling_rate,
                filter_low=self.eeg_filter_low,
                filter_high=self.eeg_filter_high
            )
            
            self.fmri_analyzer = fMRIAnalyzer(
                tr=self.fmri_tr,
                smoothing_kernel=self.fmri_smoothing_kernel
            )
            
            self.connectivity_analyzer = ConnectivityAnalyzer(
                method=self.connectivity_method
            )
            
            self.pci_calculator = PCICalculator(
                perturbation_intensity=self.pci_perturbation_intensity
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize biological processor: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for biological analysis."""
        if isinstance(data, dict):
            # Check for EEG data
            if 'eeg_data' in data:
                eeg_data = data['eeg_data']
                if isinstance(eeg_data, np.ndarray):
                    return DataValidator.validate_neural_timeseries(eeg_data)
            
            # Check for fMRI data
            if 'fmri_data' in data:
                fmri_data = data['fmri_data']
                if isinstance(fmri_data, np.ndarray):
                    return DataValidator.validate_neural_timeseries(fmri_data)
            
            # Check for connectivity matrix
            if 'connectivity_matrix' in data:
                conn_matrix = data['connectivity_matrix']
                if isinstance(conn_matrix, np.ndarray):
                    return DataValidator.validate_network_structure(conn_matrix)
            
            # Check for TMS-EEG data
            if 'tms_eeg_data' in data:
                tms_data = data['tms_eeg_data']
                return isinstance(tms_data, dict) and 'pre_stimulus' in tms_data and 'post_stimulus' in tms_data
        
        elif isinstance(data, np.ndarray):
            # Direct neural time series
            return DataValidator.validate_neural_timeseries(data)
        
        return False
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """
        Process biological data for consciousness assessment.
        
        Args:
            data: Biological data (EEG, fMRI, connectivity, TMS-EEG)
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult with biological consciousness metrics
        """
        try:
            # Parse input data
            parsed_data = self._parse_biological_data(data)
            
            # Perform biological analysis
            results = {}
            
            # EEG analysis
            if 'eeg_data' in parsed_data:
                eeg_results = self._analyze_eeg_data(parsed_data['eeg_data'])
                results.update(eeg_results)
            
            # fMRI analysis
            if 'fmri_data' in parsed_data:
                fmri_results = self._analyze_fmri_data(parsed_data['fmri_data'])
                results.update(fmri_results)
            
            # Connectivity analysis
            if 'connectivity_matrix' in parsed_data:
                connectivity_results = self._analyze_connectivity(parsed_data['connectivity_matrix'])
                results.update(connectivity_results)
            elif 'eeg_data' in parsed_data or 'fmri_data' in parsed_data:
                # Calculate connectivity from time series
                time_series = parsed_data.get('eeg_data') or parsed_data.get('fmri_data')
                connectivity_results = self._analyze_dynamic_connectivity(time_series)
                results.update(connectivity_results)
            
            # PCI analysis (TMS-EEG or perturbational analysis)
            if 'tms_eeg_data' in parsed_data:
                pci_results = self._analyze_pci_tms(parsed_data['tms_eeg_data'])
                results.update(pci_results)
            elif 'eeg_data' in parsed_data:
                # PCI proxy from spontaneous EEG
                pci_proxy_results = self._calculate_pci_proxy(parsed_data['eeg_data'])
                results.update(pci_proxy_results)
            
            # Calculate overall biological consciousness score
            bio_consciousness_score = self._calculate_biological_consciousness_score(results)
            results['biological_consciousness_score'] = bio_consciousness_score
            
            # Calculate confidence
            confidence = self._calculate_confidence(results)
            
            return ProcessingResult(
                data=results,
                metadata={
                    'eeg_sampling_rate': self.eeg_sampling_rate,
                    'fmri_tr': self.fmri_tr,
                    'connectivity_method': self.connectivity_method,
                    'processing_parameters': kwargs
                },
                timestamp=datetime.now(),
                processor_type="Biological",
                confidence=confidence
            )
            
        except Exception as e:
            return ProcessingResult(
                data=None,
                metadata={'error_details': str(e)},
                timestamp=datetime.now(),
                processor_type="Biological",
                errors=[f"Biological processing failed: {str(e)}"]
            )
    
    def _parse_biological_data(self, data: Any) -> Dict[str, Any]:
        """Parse and standardize biological data."""
        parsed = {}
        
        if isinstance(data, dict):
            # Direct dictionary input
            for key in ['eeg_data', 'fmri_data', 'connectivity_matrix', 'tms_eeg_data']:
                if key in data and isinstance(data[key], np.ndarray):
                    parsed[key] = data[key]
                elif key in data:
                    parsed[key] = data[key]  # For complex structures like TMS data
        
        elif isinstance(data, np.ndarray):
            # Determine data type based on characteristics
            if len(data.shape) == 2:
                if data.shape[0] == data.shape[1]:
                    # Square matrix - likely connectivity
                    parsed['connectivity_matrix'] = data
                else:
                    # Rectangular - likely time series
                    # Determine if EEG or fMRI based on sampling characteristics
                    if data.shape[1] > 1000:  # High temporal resolution - likely EEG
                        parsed['eeg_data'] = data
                    else:  # Lower temporal resolution - likely fMRI
                        parsed['fmri_data'] = data
        
        return parsed
    
    def _analyze_eeg_data(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Analyze EEG data for consciousness indicators."""
        results = {}
        
        # Spectral analysis
        spectral_results = self.eeg_analyzer.analyze_spectral_features(eeg_data)
        results.update(spectral_results)
        
        # Complexity measures
        complexity_results = self.eeg_analyzer.calculate_complexity_measures(eeg_data)
        results.update(complexity_results)
        
        # Microstates analysis
        microstate_results = self.eeg_analyzer.analyze_microstates(eeg_data)
        results.update(microstate_results)
        
        # Global measures
        global_results = self.eeg_analyzer.calculate_global_measures(eeg_data)
        results.update(global_results)
        
        return results
    
    def _analyze_fmri_data(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """Analyze fMRI data for consciousness indicators."""
        results = {}
        
        # Network analysis
        network_results = self.fmri_analyzer.analyze_brain_networks(fmri_data)
        results.update(network_results)
        
        # Default mode network
        dmn_results = self.fmri_analyzer.analyze_default_mode_network(fmri_data)
        results.update(dmn_results)
        
        # Global workspace analysis
        workspace_results = self.fmri_analyzer.analyze_global_workspace(fmri_data)
        results.update(workspace_results)
        
        # Criticality measures
        criticality_results = self.fmri_analyzer.calculate_criticality_measures(fmri_data)
        results.update(criticality_results)
        
        return results
    
    def _analyze_connectivity(self, connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze static connectivity for consciousness indicators."""
        results = {}
        
        # Network topology
        topology_results = self.connectivity_analyzer.analyze_network_topology(connectivity_matrix)
        results.update(topology_results)
        
        # Graph metrics
        graph_results = self.connectivity_analyzer.calculate_graph_metrics(connectivity_matrix)
        results.update(graph_results)
        
        # Community structure
        community_results = self.connectivity_analyzer.detect_community_structure(connectivity_matrix)
        results.update(community_results)
        
        return results
    
    def _analyze_dynamic_connectivity(self, time_series: np.ndarray) -> Dict[str, float]:
        """Analyze dynamic functional connectivity."""
        results = {}
        
        # Calculate dynamic connectivity
        dynamic_conn = self.connectivity_analyzer.calculate_dynamic_connectivity(time_series)
        results['dynamic_connectivity_variance'] = np.var(dynamic_conn) if dynamic_conn.size > 0 else 0.0
        
        # Flexibility and integration
        flexibility_results = self.connectivity_analyzer.calculate_network_flexibility(dynamic_conn)
        results.update(flexibility_results)
        
        # Metastability
        metastability = self.connectivity_analyzer.calculate_metastability(time_series)
        results['metastability'] = metastability
        
        return results
    
    def _analyze_pci_tms(self, tms_eeg_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze TMS-EEG data for PCI calculation."""
        results = {}
        
        if 'pre_stimulus' in tms_eeg_data and 'post_stimulus' in tms_eeg_data:
            pre_data = tms_eeg_data['pre_stimulus']
            post_data = tms_eeg_data['post_stimulus']
            
            # Calculate PCI
            pci_score = self.pci_calculator.calculate_pci_tms(pre_data, post_data)
            results['pci'] = pci_score
            
            # Spatiotemporal complexity
            st_complexity = self.pci_calculator.calculate_spatiotemporal_complexity(post_data)
            results['spatiotemporal_complexity'] = st_complexity
        
        return results
    
    def _calculate_pci_proxy(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Calculate PCI proxy from spontaneous EEG."""
        results = {}
        
        # PCI proxy based on spontaneous complexity
        pci_proxy = self.pci_calculator.calculate_pci_proxy(eeg_data)
        results['pci_proxy'] = pci_proxy
        
        # Lempel-Ziv complexity
        lz_complexity = self.pci_calculator.calculate_lempel_ziv_complexity(eeg_data)
        results['lz_complexity'] = lz_complexity
        
        return results
    
    def _calculate_biological_consciousness_score(self, results: Dict[str, float]) -> float:
        """Calculate overall biological consciousness score."""
        consciousness_indicators = []
        
        # PCI contribution (strongest indicator)
        if 'pci' in results:
            pci_contrib = min(1.0, results['pci'] / 0.5)  # Typical consciousness threshold
            consciousness_indicators.append(pci_contrib * 0.3)  # High weight
        elif 'pci_proxy' in results:
            pci_proxy_contrib = min(1.0, results['pci_proxy'] / 0.3)
            consciousness_indicators.append(pci_proxy_contrib * 0.2)
        
        # Complexity contribution
        complexity_indicators = ['lz_complexity', 'spectral_entropy', 'sample_entropy']
        complexity_scores = [results.get(ind, 0) for ind in complexity_indicators if ind in results]
        if complexity_scores:
            complexity_contrib = np.mean(complexity_scores)
            consciousness_indicators.append(complexity_contrib * 0.2)
        
        # Network integration contribution
        integration_indicators = ['global_efficiency', 'integration', 'small_world_coefficient']
        integration_scores = [results.get(ind, 0) for ind in integration_indicators if ind in results]
        if integration_scores:
            integration_contrib = np.mean(integration_scores)
            consciousness_indicators.append(integration_contrib * 0.2)
        
        # Dynamic connectivity contribution
        if 'dynamic_connectivity_variance' in results:
            # Moderate variance indicates healthy dynamics
            variance = results['dynamic_connectivity_variance']
            optimal_variance = 0.1  # Assumed optimal
            dynamics_contrib = np.exp(-abs(variance - optimal_variance) / optimal_variance)
            consciousness_indicators.append(dynamics_contrib * 0.15)
        
        # Metastability contribution
        if 'metastability' in results:
            metastability_contrib = min(1.0, results['metastability'])
            consciousness_indicators.append(metastability_contrib * 0.15)
        
        # Calculate weighted average
        if consciousness_indicators:
            return np.sum(consciousness_indicators) / len(consciousness_indicators)
        else:
            return 0.0
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in biological consciousness assessment."""
        confidence_factors = []
        
        # Data quality indicators
        if 'signal_to_noise_ratio' in results:
            snr_confidence = min(1.0, results['signal_to_noise_ratio'] / 10.0)
            confidence_factors.append(snr_confidence)
        
        # Number of completed analyses
        analysis_completeness = len([k for k in results.keys() if not k.endswith('_std')])
        max_analyses = 15  # Expected number of main analyses
        completeness_factor = min(1.0, analysis_completeness / max_analyses)
        confidence_factors.append(completeness_factor)
        
        # Consistency across measures
        consciousness_measures = [
            results.get('pci', 0),
            results.get('pci_proxy', 0), 
            results.get('lz_complexity', 0),
            results.get('global_efficiency', 0),
            results.get('metastability', 0)
        ]
        
        valid_measures = [m for m in consciousness_measures if m > 0]
        if len(valid_measures) > 1:
            consistency = 1.0 - np.std(valid_measures) / (np.mean(valid_measures) + 1e-10)
            consistency_factor = max(0.0, min(1.0, consistency))
            confidence_factors.append(consistency_factor)
        
        # Statistical significance (if available)
        if 'statistical_significance' in results:
            sig_factor = results['statistical_significance']
            confidence_factors.append(sig_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5