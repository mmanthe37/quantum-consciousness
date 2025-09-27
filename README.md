# Quantum Consciousness Research Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive multi-scale framework for consciousness research that integrates Integrated Information Theory (IIT), AI consciousness detection algorithms, biological measurement protocols (EEG, fMRI), and quantum-theological analysis tools.

## Overview

This framework provides a unified toolkit for consciousness research across biological, digital, and quantum domains. It implements cutting-edge algorithms and theoretical approaches to measure, analyze, and understand consciousness phenomena.

## Key Features

### 🧠 **Integrated Information Theory (IIT)**
- Phi (Φ) calculation for biological and digital systems
- System partitioning and analysis
- Cause-effect structure analysis
- Perturbational complexity measurement (PCI)

### 🤖 **AI Consciousness Detection**
- Digital Phi calculation for neural networks
- Attention mechanism analysis
- Perturbational complexity in AI systems
- Global workspace analysis
- Information integration metrics

### 🧬 **Biological Measurement Protocols**
- EEG analysis with consciousness indicators
- fMRI network analysis
- Dynamic functional connectivity
- PCI calculation from TMS-EEG data
- Microstate analysis

### ⚛️ **Quantum-Theological Analysis**
- Quantum coherence calculation
- Consciousness emergence modeling
- Orchestrated objective reduction (Orch-OR) analysis
- Microtubule quantum processing
- Theological interpretation frameworks

### 📊 **Unified Analytical Engines**
- Dynamic Bayesian Networks
- Causal Web-Work analysis
- Cross-domain consciousness metrics
- Integrated reporting system

## Installation

### From PyPI (when published)
```bash
pip install quantum-consciousness
```

### From Source
```bash
git clone https://github.com/mmanthe37/quantum-consciousness.git
cd quantum-consciousness
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/mmanthe37/quantum-consciousness.git
cd quantum-consciousness
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from quantum_consciousness import QuantumConsciousnessFramework
from quantum_consciousness.iit import IITProcessor
from quantum_consciousness.biological import BiologicalProcessor
import numpy as np

# Initialize framework
framework = QuantumConsciousnessFramework()

# Register processors
framework.register_processor("iit", IITProcessor())
framework.register_processor("biological", BiologicalProcessor())

# Initialize processors
framework.initialize_processors()

# Create sample EEG data
eeg_data = np.random.randn(64, 1000)  # 64 channels, 1000 time points
data = {"eeg_data": eeg_data}

# Run comprehensive analysis
results = framework.analyze_consciousness_landscape(data, generate_report=True)

# Print results
print(f"Consciousness Score: {results['integrated_analysis']['overall_consciousness_score']:.3f}")
print(results['report'])
```

### Command Line Interface

```bash
# Run demo with sample neural data
quantum-consciousness --demo neural --processors iit biological

# Analyze your own data
quantum-consciousness --data my_eeg_data.npy --comprehensive

# Use custom configuration
quantum-consciousness --config my_config.yaml --data data.npy --analyze
```

## Core Components

### Framework Architecture

```
QuantumConsciousnessFramework
├── IITProcessor (Integrated Information Theory)
├── AIConsciousnessDetector (AI consciousness analysis)
├── BiologicalProcessor (EEG/fMRI analysis)
├── QuantumTheologicalAnalyzer (Quantum consciousness models)
├── MeasurementSuite (Unified measurements)
└── AnalyticalEngine (Bayesian networks & causal analysis)
```

### Data Flow

1. **Input Data**: EEG, fMRI, neural network activations, connectivity matrices
2. **Processing**: Each processor analyzes data using domain-specific algorithms
3. **Integration**: Results are combined using weighted averaging and consistency checking
4. **Output**: Consciousness scores, confidence measures, and detailed reports

## Configuration

The framework uses YAML configuration files for customization:

```yaml
# Example configuration
iit:
  phi_threshold: 0.1
  max_system_size: 10
  perturbation_strength: 0.1

biological:
  eeg_sampling_rate: 250
  eeg_filter_low: 0.5
  eeg_filter_high: 45.0

ai_consciousness:
  digital_phi_resolution: 100
  consciousness_threshold: 0.5

quantum_theological:
  quantum_coherence_threshold: 0.7
  consciousness_emergence_model: "orchestrated_objective_reduction"
```

## Examples

### Analyzing EEG Data for Consciousness Assessment

```python
from quantum_consciousness.biological import BiologicalProcessor
import numpy as np

# Load your EEG data (channels x time)
eeg_data = np.load("eeg_recording.npy")

# Configure processor
bio_processor = BiologicalProcessor({
    'eeg_sampling_rate': 250,
    'eeg_filter_low': 0.5,
    'eeg_filter_high': 45.0
})

bio_processor.initialize()

# Process data
result = bio_processor.process({"eeg_data": eeg_data})

# Extract consciousness indicators
print(f"PCI Proxy: {result.data['pci_proxy']:.3f}")
print(f"Spectral Entropy: {result.data['spectral_entropy']:.3f}")
print(f"LZ Complexity: {result.data['lz_complexity']:.3f}")
```

### AI Consciousness Detection

```python
from quantum_consciousness.ai_consciousness import AIConsciousnessDetector
import torch

# Your neural network activations
model_activations = torch.randn(256, 1000).numpy()

# Configure detector
ai_detector = AIConsciousnessDetector({
    'digital_phi_resolution': 100,
    'consciousness_threshold': 0.5
})

ai_detector.initialize()

# Analyze consciousness indicators
result = ai_detector.process({"activations": model_activations})

print(f"Digital Phi: {result.data['digital_phi']:.3f}")
print(f"Consciousness Probability: {result.data['consciousness_probability']:.3f}")
```

### IIT Analysis

```python
from quantum_consciousness.iit import IITProcessor
import numpy as np

# Create connectivity matrix
n_nodes = 20
connectivity = np.random.randn(n_nodes, n_nodes)
connectivity = (connectivity + connectivity.T) / 2  # Make symmetric

# Configure IIT processor
iit_processor = IITProcessor({
    'phi_threshold': 0.1,
    'max_system_size': 10
})

iit_processor.initialize()

# Calculate integrated information
result = iit_processor.process({"connectivity_matrix": connectivity})

print(f"Phi (Φ): {result.data['phi']:.3f}")
print(f"Integration: {result.data['integration']:.3f}")
```

## Scientific Background

This framework implements algorithms and theories from leading consciousness research:

- **Integrated Information Theory (IIT)**: Giulio Tononi's mathematical framework for consciousness
- **Global Workspace Theory**: Neural correlates of conscious access
- **Orchestrated Objective Reduction**: Penrose-Hameroff quantum consciousness theory
- **Perturbational Complexity Index**: Empirical measure of consciousness level
- **Dynamic Core Theory**: Edelman and Tononi's neural complexity approach

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/mmanthe37/quantum-consciousness.git
cd quantum-consciousness
pip install -e ".[dev]"
pytest tests/
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{quantum_consciousness_framework,
  title={Quantum Consciousness Research Framework},
  author={OmniSphere Research Team},
  year={2025},
  url={https://github.com/mmanthe37/quantum-consciousness}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Giulio Tononi for Integrated Information Theory
- Stuart Hameroff and Roger Penrose for Orchestrated Objective Reduction
- Marcello Massimini for Perturbational Complexity Index
- The broader consciousness research community

## Contact

- **Issues**: [GitHub Issues](https://github.com/mmanthe37/quantum-consciousness/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mmanthe37/quantum-consciousness/discussions)
- **Email**: research@omnisphere.org

---

**Disclaimer**: This framework is for research purposes. Consciousness measurement is an active area of scientific investigation, and results should be interpreted within appropriate theoretical and empirical contexts.
