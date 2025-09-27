"""
Command-line interface for quantum consciousness framework.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from .core.framework import QuantumConsciousnessFramework
from .core.config import FrameworkConfig
from .iit import IITProcessor
from .ai_consciousness import AIConsciousnessDetector
from .biological import BiologicalProcessor
from .quantum_theological import QuantumTheologicalAnalyzer
from .measurement import MeasurementSuite
from .analytical import AnalyticalEngine


def create_sample_data(data_type: str = "neural", size: int = 100):
    """Create sample data for testing."""
    if data_type == "neural":
        # Neural time series (channels x time)
        n_channels = 64
        n_timepoints = size * 10
        data = np.random.randn(n_channels, n_timepoints)
        
        # Add some structure
        for i in range(min(10, n_channels)):
            data[i] = np.sin(2 * np.pi * 0.01 * np.arange(n_timepoints)) + 0.5 * np.random.randn(n_timepoints)
        
        return {"eeg_data": data}
    
    elif data_type == "connectivity":
        # Connectivity matrix
        n_nodes = 50
        connectivity = np.random.randn(n_nodes, n_nodes)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity, 1.0)
        return {"connectivity_matrix": connectivity}
    
    elif data_type == "ai":
        # AI activations
        n_units = 256
        n_samples = size
        activations = np.random.randn(n_units, n_samples)
        return {"activations": activations}
    
    else:
        # Generic data
        return np.random.randn(10, size)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Quantum Consciousness Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quantum-consciousness --demo neural --processors iit biological
  quantum-consciousness --config config.yaml --data data.npy
  quantum-consciousness --analyze --comprehensive
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to input data file (.npy, .mat, .csv)"
    )
    
    parser.add_argument(
        "--processors", "-p",
        nargs="+",
        choices=["iit", "ai", "biological", "quantum", "measurement", "analytical"],
        default=["iit", "biological"],
        help="Processors to use for analysis"
    )
    
    parser.add_argument(
        "--demo",
        type=str,
        choices=["neural", "connectivity", "ai"],
        help="Run demo with sample data"
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive analysis with all processors"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run full analysis pipeline"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = FrameworkConfig.from_file(args.config)
    else:
        config = FrameworkConfig()
    
    if args.verbose:
        config.log_level = "DEBUG"
    
    # Initialize framework
    framework = QuantumConsciousnessFramework(config)
    
    # Register processors
    processor_map = {
        "iit": IITProcessor,
        "ai": AIConsciousnessDetector,
        "biological": BiologicalProcessor,
        "quantum": QuantumTheologicalAnalyzer,
        "measurement": MeasurementSuite,
        "analytical": AnalyticalEngine
    }
    
    if args.comprehensive:
        selected_processors = list(processor_map.keys())
    else:
        selected_processors = args.processors
    
    for processor_name in selected_processors:
        if processor_name in processor_map:
            processor_class = processor_map[processor_name]
            processor_instance = processor_class(config.__dict__.get(processor_name, {}))
            framework.register_processor(processor_name, processor_instance)
    
    # Initialize processors
    init_results = framework.initialize_processors()
    
    if args.verbose:
        print("Processor initialization results:")
        for name, success in init_results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
    
    # Load or create data
    if args.demo:
        data = create_sample_data(args.demo)
        print(f"Created sample {args.demo} data")
    elif args.data:
        data_path = Path(args.data)
        if data_path.suffix == ".npy":
            data = np.load(data_path)
        else:
            print(f"Unsupported data format: {data_path.suffix}")
            sys.exit(1)
    else:
        print("No data provided. Use --demo or --data")
        sys.exit(1)
    
    # Run analysis
    if args.analyze or args.demo or args.comprehensive:
        print("Running consciousness analysis...")
        
        try:
            # Run comprehensive analysis
            analysis_results = framework.analyze_consciousness_landscape(
                data,
                generate_report=True
            )
            
            # Print results
            if args.verbose:
                print("\n" + "="*50)
                print("CONSCIOUSNESS ANALYSIS RESULTS")
                print("="*50)
                
                # Print individual processor results
                for processor_name, result in analysis_results["individual_results"].items():
                    print(f"\n{processor_name.upper()}:")
                    if result["status"] == "success":
                        print(f"  Status: ✓ SUCCESS")
                        if "confidence" in result and result["confidence"]:
                            print(f"  Confidence: {result['confidence']:.3f}")
                    else:
                        print(f"  Status: ✗ ERROR")
                        if "errors" in result:
                            for error in result["errors"]:
                                print(f"  Error: {error}")
                
                # Print integrated analysis
                print(f"\nINTEGRATED ANALYSIS:")
                integrated = analysis_results["integrated_analysis"]
                print(f"  Overall Consciousness Score: {integrated['overall_consciousness_score']:.3f}")
                print(f"  Confidence-Weighted Average: {integrated['confidence_weighted_average']:.3f}")
            
            # Print report if generated
            if "report" in analysis_results and not args.verbose:
                print("\n" + analysis_results["report"])
            
            # Save results if requested
            if args.output:
                output_path = Path(args.output)
                if output_path.suffix == ".npy":
                    np.save(output_path, analysis_results)
                else:
                    import json
                    with open(output_path, 'w') as f:
                        # Convert numpy arrays to lists for JSON serialization
                        json_results = {}
                        for key, value in analysis_results.items():
                            if isinstance(value, np.ndarray):
                                json_results[key] = value.tolist()
                            elif isinstance(value, dict):
                                json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                   for k, v in value.items()}
                            else:
                                json_results[key] = value
                        json.dump(json_results, f, indent=2)
                
                print(f"\nResults saved to: {output_path}")
        
        except Exception as e:
            print(f"Analysis failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    else:
        print("No analysis requested. Use --analyze, --demo, or --comprehensive")


if __name__ == "__main__":
    main()