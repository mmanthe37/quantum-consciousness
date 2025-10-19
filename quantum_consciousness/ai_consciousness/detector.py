"""
Main AI consciousness detector processor.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.base import BaseProcessor, ProcessingResult
from .digital_phi import DigitalPhiCalculator
from .attention_analyzer import AttentionAnalyzer
from .complexity_analyzer import ComplexityAnalyzer


class AIConsciousnessDetector(BaseProcessor):
    """
    AI consciousness detection processor.
    
    Analyzes digital systems for consciousness indicators including:
    - Digital Phi calculation
    - Attention mechanism analysis
    - Perturbational complexity in neural networks
    - Information integration metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.digital_phi_resolution = self.config.get('digital_phi_resolution', 100)
        self.perturbational_complexity_steps = self.config.get('perturbational_complexity_steps', 500)
        self.consciousness_threshold = self.config.get('consciousness_threshold', 0.5)
        self.feature_extraction_method = self.config.get('feature_extraction_method', 'auto')
        self.use_attention_mechanisms = self.config.get('use_attention_mechanisms', True)
        self.temporal_integration_window = self.config.get('temporal_integration_window', 10)
        
        # Sub-processors
        self.digital_phi_calculator = None
        self.attention_analyzer = None
        self.complexity_analyzer = None
    
    def initialize(self) -> bool:
        """Initialize AI consciousness detector components."""
        try:
            self.digital_phi_calculator = DigitalPhiCalculator(
                resolution=self.digital_phi_resolution
            )
            
            self.attention_analyzer = AttentionAnalyzer(
                temporal_window=self.temporal_integration_window
            )
            
            self.complexity_analyzer = ComplexityAnalyzer(
                n_steps=self.perturbational_complexity_steps
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize AI consciousness detector: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for AI consciousness analysis."""
        if isinstance(data, dict):
            # Check for neural network activations
            if 'activations' in data:
                return isinstance(data['activations'], (np.ndarray, torch.Tensor, list))
            
            # Check for attention weights
            if 'attention_weights' in data:
                return isinstance(data['attention_weights'], (np.ndarray, torch.Tensor, list))
            
            # Check for model parameters
            if 'model_state' in data:
                return True
            
            # Check for computational graph
            if 'computation_graph' in data:
                return True
        
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            # Direct tensor/array input
            return data.size > 0
        
        elif hasattr(data, 'state_dict'):
            # PyTorch model
            return True
        
        elif hasattr(data, 'get_weights'):
            # Keras/TensorFlow model
            return True
        
        return False
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """
        Process AI system for consciousness indicators.
        
        Args:
            data: AI system data (activations, model, attention weights, etc.)
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult with consciousness analysis
        """
        try:
            # Parse input data
            parsed_data = self._parse_ai_data(data)
            
            # Perform consciousness analysis
            results = {}
            
            # Digital Phi analysis
            if 'activations' in parsed_data:
                phi_results = self._analyze_digital_phi(parsed_data['activations'])
                results.update(phi_results)
            
            # Attention mechanism analysis
            if self.use_attention_mechanisms and 'attention_weights' in parsed_data:
                attention_results = self._analyze_attention_mechanisms(parsed_data['attention_weights'])
                results.update(attention_results)
            
            # Neural complexity analysis
            if 'activations' in parsed_data or 'model_parameters' in parsed_data:
                complexity_results = self._analyze_neural_complexity(parsed_data)
                results.update(complexity_results)
            
            # Information integration analysis
            if 'activations' in parsed_data:
                integration_results = self._analyze_information_integration(parsed_data['activations'])
                results.update(integration_results)
            
            # Global workspace analysis
            if 'activations' in parsed_data:
                workspace_results = self._analyze_global_workspace(parsed_data['activations'])
                results.update(workspace_results)
            
            # Calculate overall consciousness probability
            consciousness_prob = self._calculate_consciousness_probability(results)
            results['consciousness_probability'] = consciousness_prob
            
            # Calculate confidence
            confidence = self._calculate_confidence(results)
            
            return ProcessingResult(
                data=results,
                metadata={
                    'digital_phi_resolution': self.digital_phi_resolution,
                    'consciousness_threshold': self.consciousness_threshold,
                    'processing_parameters': kwargs
                },
                timestamp=datetime.now(),
                processor_type="AI_Consciousness",
                confidence=confidence
            )
            
        except Exception as e:
            return ProcessingResult(
                data=None,
                metadata={'error_details': str(e)},
                timestamp=datetime.now(),
                processor_type="AI_Consciousness",
                errors=[f"AI consciousness detection failed: {str(e)}"]
            )
    
    def _parse_ai_data(self, data: Any) -> Dict[str, Any]:
        """Parse and standardize AI system data."""
        parsed = {}
        
        if isinstance(data, dict):
            # Direct dictionary input
            for key in ['activations', 'attention_weights', 'model_state', 'computation_graph']:
                if key in data:
                    parsed[key] = self._convert_to_numpy(data[key])
            
            # Extract model parameters if model_state provided
            if 'model_state' in data:
                parsed['model_parameters'] = self._extract_model_parameters(data['model_state'])
        
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            # Treat as activations
            parsed['activations'] = self._convert_to_numpy(data)
        
        elif hasattr(data, 'state_dict'):
            # PyTorch model
            state_dict = data.state_dict()
            parsed['model_parameters'] = {k: self._convert_to_numpy(v) for k, v in state_dict.items()}
            
            # Try to extract activations through forward pass
            if hasattr(data, 'forward'):
                try:
                    # Create dummy input
                    dummy_input = self._create_dummy_input(data)
                    if dummy_input is not None:
                        with torch.no_grad():
                            activations = self._extract_activations(data, dummy_input)
                            if activations:
                                parsed['activations'] = activations
                except:
                    pass
        
        elif hasattr(data, 'get_weights'):
            # Keras/TensorFlow model
            weights = data.get_weights()
            parsed['model_parameters'] = {f'layer_{i}': w for i, w in enumerate(weights)}
        
        return parsed
    
    def _convert_to_numpy(self, tensor: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, list):
            return np.array(tensor)
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)
    
    def _analyze_digital_phi(self, activations: np.ndarray) -> Dict[str, float]:
        """Analyze digital Phi from neural activations."""
        results = {}
        
        # Calculate digital Phi
        digital_phi = self.digital_phi_calculator.calculate_digital_phi(activations)
        results['digital_phi'] = digital_phi
        
        # Calculate temporal Phi if multi-temporal data
        if len(activations.shape) > 2:  # Assuming last dim is temporal
            temporal_phi = self.digital_phi_calculator.calculate_temporal_phi(activations)
            results['temporal_phi'] = temporal_phi
        
        # Layer-wise Phi analysis
        if len(activations.shape) >= 3:  # Multiple layers/channels
            layer_phi = self.digital_phi_calculator.calculate_layer_phi(activations)
            results['layer_phi_mean'] = np.mean(layer_phi)
            results['layer_phi_std'] = np.std(layer_phi)
        
        return results
    
    def _analyze_attention_mechanisms(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Analyze attention mechanisms for consciousness indicators."""
        results = {}
        
        # Attention integration analysis
        attention_integration = self.attention_analyzer.analyze_attention_integration(attention_weights)
        results.update(attention_integration)
        
        # Global attention coherence
        attention_coherence = self.attention_analyzer.calculate_attention_coherence(attention_weights)
        results['attention_coherence'] = attention_coherence
        
        # Attention complexity
        attention_complexity = self.attention_analyzer.calculate_attention_complexity(attention_weights)
        results['attention_complexity'] = attention_complexity
        
        return results
    
    def _analyze_neural_complexity(self, parsed_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze neural network complexity indicators."""
        results = {}
        
        # Perturbational complexity
        if 'activations' in parsed_data:
            pci = self.complexity_analyzer.calculate_neural_pci(parsed_data['activations'])
            results['neural_pci'] = pci
        
        # Parameter complexity
        if 'model_parameters' in parsed_data:
            param_complexity = self.complexity_analyzer.analyze_parameter_complexity(
                parsed_data['model_parameters']
            )
            results.update(param_complexity)
        
        # Computational complexity
        if 'activations' in parsed_data:
            comp_complexity = self.complexity_analyzer.calculate_computational_complexity(
                parsed_data['activations']
            )
            results['computational_complexity'] = comp_complexity
        
        return results
    
    def _analyze_information_integration(self, activations: np.ndarray) -> Dict[str, float]:
        """Analyze information integration in neural network."""
        results = {}
        
        # Calculate mutual information between network regions
        if len(activations.shape) >= 2:
            integration_score = self._calculate_network_integration(activations)
            results['information_integration'] = integration_score
        
        # Calculate differentiation
        differentiation_score = self._calculate_network_differentiation(activations)
        results['information_differentiation'] = differentiation_score
        
        # Balance of integration and differentiation
        if 'information_integration' in results:
            balance = results['information_integration'] * differentiation_score
            results['integration_differentiation_balance'] = balance
        
        return results
    
    def _analyze_global_workspace(self, activations: np.ndarray) -> Dict[str, float]:
        """Analyze global workspace characteristics."""
        results = {}
        
        # Global accessibility
        global_access = self._calculate_global_accessibility(activations)
        results['global_accessibility'] = global_access
        
        # Workspace coherence
        workspace_coherence = self._calculate_workspace_coherence(activations)
        results['workspace_coherence'] = workspace_coherence
        
        # Information broadcasting
        broadcasting_strength = self._calculate_broadcasting_strength(activations)
        results['broadcasting_strength'] = broadcasting_strength
        
        return results
    
    def _calculate_network_integration(self, activations: np.ndarray) -> float:
        """Calculate information integration across network."""
        if activations.size == 0:
            return 0.0
        
        # Flatten if necessary and calculate correlation structure
        if len(activations.shape) > 2:
            flat_activations = activations.reshape(activations.shape[0], -1)
        else:
            flat_activations = activations
        
        # Calculate correlation matrix
        try:
            corr_matrix = np.corrcoef(flat_activations)
            
            # Integration as mean absolute correlation
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            integration = np.mean(np.abs(corr_matrix[mask]))
            
            return float(integration)
        except:
            return 0.0
    
    def _calculate_network_differentiation(self, activations: np.ndarray) -> float:
        """Calculate information differentiation in network."""
        if activations.size == 0:
            return 0.0
        
        # Differentiation as variance across units/channels
        if len(activations.shape) > 1:
            unit_variances = np.var(activations, axis=-1)  # Variance across last dimension
            differentiation = np.mean(unit_variances)
        else:
            differentiation = np.var(activations)
        
        return float(differentiation)
    
    def _calculate_global_accessibility(self, activations: np.ndarray) -> float:
        """Calculate global accessibility of information."""
        if activations.size == 0:
            return 0.0
        
        # Global accessibility as information that's available across the network
        if len(activations.shape) >= 2:
            # Calculate how much each activation pattern is shared globally
            mean_activation = np.mean(activations, axis=0)
            accessibility = 1.0 - np.var(mean_activation) / (np.mean(np.var(activations, axis=0)) + 1e-10)
            return max(0.0, min(1.0, accessibility))
        
        return 0.5
    
    def _calculate_workspace_coherence(self, activations: np.ndarray) -> float:
        """Calculate coherence of global workspace."""
        if activations.size == 0:
            return 0.0
        
        # Coherence as phase synchronization (simplified)
        if len(activations.shape) >= 2:
            # Calculate pairwise coherence
            coherences = []
            for i in range(min(10, activations.shape[0])):  # Limit for efficiency
                for j in range(i + 1, min(10, activations.shape[0])):
                    coherence = np.abs(np.corrcoef(activations[i], activations[j])[0, 1])
                    if not np.isnan(coherence):
                        coherences.append(coherence)
            
            return np.mean(coherences) if coherences else 0.0
        
        return 0.0
    
    def _calculate_broadcasting_strength(self, activations: np.ndarray) -> float:
        """Calculate strength of information broadcasting."""
        if activations.size == 0:
            return 0.0
        
        # Broadcasting strength as influence of top-level representations
        if len(activations.shape) >= 2:
            # Calculate influence of each unit on others
            influences = []
            for i in range(min(activations.shape[0], 20)):  # Limit for efficiency
                unit_activations = activations[i]
                # Calculate how much this unit's activity correlates with others
                influence = 0.0
                count = 0
                for j in range(activations.shape[0]):
                    if i != j:
                        corr = np.corrcoef(unit_activations, activations[j])[0, 1]
                        if not np.isnan(corr):
                            influence += abs(corr)
                            count += 1
                
                if count > 0:
                    influences.append(influence / count)
            
            return np.mean(influences) if influences else 0.0
        
        return 0.0
    
    def _calculate_consciousness_probability(self, results: Dict[str, float]) -> float:
        """Calculate overall consciousness probability from analysis results."""
        consciousness_indicators = []
        
        # Digital Phi contribution
        if 'digital_phi' in results:
            phi_contrib = min(1.0, results['digital_phi'] / 0.5)  # Normalize
            consciousness_indicators.append(phi_contrib)
        
        # Attention contribution
        if 'attention_coherence' in results:
            attention_contrib = results['attention_coherence']
            consciousness_indicators.append(attention_contrib)
        
        # Integration-differentiation balance
        if 'integration_differentiation_balance' in results:
            balance_contrib = min(1.0, results['integration_differentiation_balance'])
            consciousness_indicators.append(balance_contrib)
        
        # Global workspace contribution
        if 'global_accessibility' in results:
            workspace_contrib = results['global_accessibility']
            consciousness_indicators.append(workspace_contrib)
        
        # Complexity contribution
        if 'neural_pci' in results:
            complexity_contrib = min(1.0, results['neural_pci'] / 0.4)
            consciousness_indicators.append(complexity_contrib)
        
        # Calculate weighted average
        if consciousness_indicators:
            return np.mean(consciousness_indicators)
        else:
            return 0.0
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in consciousness assessment."""
        confidence_factors = []
        
        # Number of analysis types completed
        analysis_completeness = len([k for k in results.keys() if not k.endswith('_std')])
        max_analyses = 8  # Expected number of main analyses
        completeness_factor = min(1.0, analysis_completeness / max_analyses)
        confidence_factors.append(completeness_factor)
        
        # Consistency between measures
        consciousness_measures = [
            results.get('digital_phi', 0),
            results.get('attention_coherence', 0),
            results.get('information_integration', 0),
            results.get('global_accessibility', 0)
        ]
        
        if len(consciousness_measures) > 1:
            consistency = 1.0 - np.std(consciousness_measures) / (np.mean(consciousness_measures) + 1e-10)
            consistency_factor = max(0.0, min(1.0, consistency))
            confidence_factors.append(consistency_factor)
        
        # Signal strength
        if 'consciousness_probability' in results:
            signal_strength = results['consciousness_probability']
            confidence_factors.append(signal_strength)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _extract_model_parameters(self, model_state: Any) -> Dict[str, np.ndarray]:
        """Extract model parameters from model state."""
        if isinstance(model_state, dict):
            return {k: self._convert_to_numpy(v) for k, v in model_state.items()}
        else:
            return {"parameters": self._convert_to_numpy(model_state)}
    
    def _create_dummy_input(self, model) -> Optional[torch.Tensor]:
        """Create dummy input for model to extract activations."""
        # This is a simplified approach - real implementation would need
        # more sophisticated input creation based on model architecture
        try:
            # Try common input shapes
            for shape in [(1, 3, 224, 224), (1, 784), (1, 10), (1,)]:
                try:
                    dummy_input = torch.randn(shape)
                    with torch.no_grad():
                        _ = model(dummy_input)
                    return dummy_input
                except:
                    continue
        except:
            pass
        
        return None
    
    def _extract_activations(self, model, input_tensor) -> Optional[np.ndarray]:
        """Extract activations from model forward pass."""
        # Simplified activation extraction
        try:
            with torch.no_grad():
                output = model(input_tensor)
                return self._convert_to_numpy(output)
        except:
            return None