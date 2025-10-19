#!/usr/bin/env python3
"""
Basic usage example for the Quantum Consciousness Research Framework.
"""

import numpy as np


def main():
    """Main example demonstrating framework usage."""
    print("Quantum Consciousness Research Framework - Basic Usage Example")
    print("=" * 60)
    
    try:
        from quantum_consciousness import QuantumConsciousnessFramework
        from quantum_consciousness.iit import IITProcessor
        from quantum_consciousness.biological import BiologicalProcessor
        
        # 1. Initialize the framework
        print("\n1. Initializing framework...")
        framework = QuantumConsciousnessFramework()
        
        # 2. Register processors
        print("2. Registering processors...")
        
        # IIT processor for integrated information analysis
        iit_processor = IITProcessor({
            'phi_threshold': 0.1,
            'max_system_size': 10,
            'perturbation_strength': 0.1
        })
        framework.register_processor("iit", iit_processor)
        
        # Biological processor for EEG analysis
        bio_processor = BiologicalProcessor({
            'eeg_sampling_rate': 250,
            'eeg_filter_low': 0.5,
            'eeg_filter_high': 45.0
        })
        framework.register_processor("biological", bio_processor)
        
        # 3. Initialize processors
        print("3. Initializing processors...")
        init_results = framework.initialize_processors()
        
        for processor_name, success in init_results.items():
            status = "✓" if success else "✗"
            print(f"   {status} {processor_name}")
        
        # 4. Create sample EEG data
        print("\n4. Creating sample EEG data...")
        n_channels = 64
        n_timepoints = 1000
        eeg_data = np.random.randn(n_channels, n_timepoints)
        
        # Add some structured oscillations
        time = np.arange(n_timepoints) / 250.0  # 250 Hz sampling rate
        for ch in range(min(10, n_channels)):
            alpha_freq = 10  # Hz
            eeg_data[ch] += np.sin(2 * np.pi * alpha_freq * time)
        
        biological_data = {"eeg_data": eeg_data}
        
        # 5. Run comprehensive consciousness analysis
        print("5. Running consciousness analysis...")
        comprehensive_results = framework.analyze_consciousness_landscape(
            biological_data,
            generate_report=True
        )
        
        print(f"   Overall Consciousness Score: {comprehensive_results['integrated_analysis']['overall_consciousness_score']:.3f}")
        
        # Print summary report
        if 'report' in comprehensive_results:
            print("\n" + comprehensive_results['report'])
        
        print("\nExample completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install the package: pip install -e .")
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()