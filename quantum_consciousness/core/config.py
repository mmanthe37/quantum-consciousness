"""
Configuration management for the quantum consciousness framework.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path


@dataclass
class IITConfig:
    """Configuration for Integrated Information Theory processing."""
    
    phi_threshold: float = 0.1
    max_system_size: int = 10
    perturbation_strength: float = 0.1
    integration_steps: int = 1000
    use_gaussian_perturbation: bool = True
    calculate_phi_max: bool = True


@dataclass
class AIConsciousnessConfig:
    """Configuration for AI consciousness detection."""
    
    digital_phi_resolution: int = 100
    perturbational_complexity_steps: int = 500
    consciousness_threshold: float = 0.5
    feature_extraction_method: str = "auto"
    use_attention_mechanisms: bool = True
    temporal_integration_window: int = 10


@dataclass
class BiologicalConfig:
    """Configuration for biological measurement processing."""
    
    eeg_sampling_rate: int = 250
    eeg_filter_low: float = 0.5
    eeg_filter_high: float = 45.0
    fmri_tr: float = 2.0
    fmri_smoothing_kernel: float = 6.0
    pci_perturbation_intensity: float = 100.0
    connectivity_method: str = "pearson"


@dataclass 
class QuantumTheologicalConfig:
    """Configuration for quantum-theological analysis."""
    
    quantum_coherence_threshold: float = 0.7
    theological_interpretation_depth: int = 3
    consciousness_emergence_model: str = "orchestrated_objective_reduction"
    quantum_information_integration: bool = True
    microtubule_modeling: bool = False


@dataclass
class AnalyticalConfig:
    """Configuration for analytical engines."""
    
    bayesian_mcmc_samples: int = 5000
    bayesian_burnin: int = 1000
    causal_discovery_method: str = "pc"
    causal_significance_threshold: float = 0.05
    network_analysis_metrics: List[str] = field(default_factory=lambda: [
        "clustering", "efficiency", "modularity", "small_world"
    ])


@dataclass
class FrameworkConfig:
    """Main framework configuration."""
    
    # Sub-configurations
    iit: IITConfig = field(default_factory=IITConfig)
    ai_consciousness: AIConsciousnessConfig = field(default_factory=AIConsciousnessConfig)
    biological: BiologicalConfig = field(default_factory=BiologicalConfig)
    quantum_theological: QuantumTheologicalConfig = field(default_factory=QuantumTheologicalConfig)
    analytical: AnalyticalConfig = field(default_factory=AnalyticalConfig)
    
    # General settings
    parallel_processing: bool = True
    max_workers: int = 4
    cache_results: bool = True
    cache_directory: str = "./cache"
    log_level: str = "INFO"
    random_seed: Optional[int] = 42
    
    # Output settings
    save_intermediate_results: bool = True
    output_format: str = "hdf5"  # "hdf5", "pickle", "json"
    compression: bool = True
    
    @classmethod
    def from_file(cls, config_path: str) -> 'FrameworkConfig':
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FrameworkConfig':
        """Create configuration from dictionary."""
        # Extract sub-configurations
        iit_config = IITConfig(**config_dict.get('iit', {}))
        ai_config = AIConsciousnessConfig(**config_dict.get('ai_consciousness', {}))
        bio_config = BiologicalConfig(**config_dict.get('biological', {}))
        qt_config = QuantumTheologicalConfig(**config_dict.get('quantum_theological', {}))
        anal_config = AnalyticalConfig(**config_dict.get('analytical', {}))
        
        # Extract general settings
        general_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['iit', 'ai_consciousness', 'biological', 
                                     'quantum_theological', 'analytical']}
        
        return cls(
            iit=iit_config,
            ai_consciousness=ai_config,
            biological=bio_config,
            quantum_theological=qt_config,
            analytical=anal_config,
            **general_settings
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'iit': self.iit.__dict__,
            'ai_consciousness': self.ai_consciousness.__dict__,
            'biological': self.biological.__dict__,
            'quantum_theological': self.quantum_theological.__dict__,
            'analytical': self.analytical.__dict__,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'cache_results': self.cache_results,
            'cache_directory': self.cache_directory,
            'log_level': self.log_level,
            'random_seed': self.random_seed,
            'save_intermediate_results': self.save_intermediate_results,
            'output_format': self.output_format,
            'compression': self.compression,
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate IIT config
        if self.iit.phi_threshold < 0:
            issues.append("IIT phi_threshold must be non-negative")
        if self.iit.max_system_size < 1:
            issues.append("IIT max_system_size must be positive")
        
        # Validate AI consciousness config
        if self.ai_consciousness.digital_phi_resolution < 10:
            issues.append("AI digital_phi_resolution should be at least 10")
        if not 0 <= self.ai_consciousness.consciousness_threshold <= 1:
            issues.append("AI consciousness_threshold must be between 0 and 1")
        
        # Validate biological config
        if self.biological.eeg_sampling_rate < 1:
            issues.append("Biological EEG sampling rate must be positive")
        if self.biological.eeg_filter_low >= self.biological.eeg_filter_high:
            issues.append("Biological EEG filter_low must be less than filter_high")
        
        # Validate general settings
        if self.max_workers < 1:
            issues.append("max_workers must be positive")
        if self.output_format not in ['hdf5', 'pickle', 'json']:
            issues.append("output_format must be 'hdf5', 'pickle', or 'json'")
        
        return issues


def create_default_config() -> FrameworkConfig:
    """Create a default configuration with reasonable values."""
    return FrameworkConfig()


def load_config(config_path: Optional[str] = None) -> FrameworkConfig:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        return FrameworkConfig.from_file(config_path)
    else:
        return create_default_config()