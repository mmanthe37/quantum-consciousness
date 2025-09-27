"""
Main quantum consciousness research framework.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import json
import h5py

from .base import BaseProcessor, ProcessingResult
from .config import FrameworkConfig, load_config


class QuantumConsciousnessFramework:
    """
    Main framework for integrated quantum consciousness research.
    
    Coordinates multiple analysis modules including IIT, AI consciousness detection,
    biological measurements, and quantum-theological analysis.
    """
    
    def __init__(self, config: Optional[Union[FrameworkConfig, str, Dict[str, Any]]] = None):
        """
        Initialize the quantum consciousness framework.
        
        Args:
            config: Configuration object, file path, or dictionary
        """
        # Load configuration
        if isinstance(config, str):
            self.config = FrameworkConfig.from_file(config)
        elif isinstance(config, dict):
            self.config = FrameworkConfig.from_dict(config)
        elif isinstance(config, FrameworkConfig):
            self.config = config
        else:
            self.config = load_config()
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ValueError(f"Configuration validation failed: {config_issues}")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize processors
        self.processors: Dict[str, BaseProcessor] = {}
        self._is_initialized = False
        
        # Setup cache directory
        cache_path = Path(self.config.cache_directory)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Quantum Consciousness Framework initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def register_processor(self, name: str, processor: BaseProcessor) -> None:
        """Register a new processor with the framework."""
        if not isinstance(processor, BaseProcessor):
            raise TypeError("Processor must inherit from BaseProcessor")
        
        self.processors[name] = processor
        self.logger.info(f"Registered processor: {name}")
    
    def initialize_processors(self) -> Dict[str, bool]:
        """Initialize all registered processors."""
        initialization_results = {}
        
        for name, processor in self.processors.items():
            try:
                success = processor.initialize()
                initialization_results[name] = success
                if success:
                    self.logger.info(f"Successfully initialized processor: {name}")
                else:
                    self.logger.warning(f"Failed to initialize processor: {name}")
            except Exception as e:
                self.logger.error(f"Error initializing processor {name}: {str(e)}")
                initialization_results[name] = False
        
        self._is_initialized = all(initialization_results.values())
        return initialization_results
    
    def process_data(self, 
                    data: Any, 
                    processors: Optional[List[str]] = None,
                    parallel: Optional[bool] = None) -> Dict[str, ProcessingResult]:
        """
        Process data through specified processors.
        
        Args:
            data: Input data to process
            processors: List of processor names to use (None for all)
            parallel: Whether to use parallel processing (None for config default)
            
        Returns:
            Dictionary mapping processor names to their results
        """
        if not self._is_initialized:
            raise RuntimeError("Framework not initialized. Call initialize_processors() first.")
        
        # Determine which processors to use
        if processors is None:
            processors = list(self.processors.keys())
        
        # Validate processor names
        invalid_processors = set(processors) - set(self.processors.keys())
        if invalid_processors:
            raise ValueError(f"Unknown processors: {invalid_processors}")
        
        # Determine parallel processing
        use_parallel = parallel if parallel is not None else self.config.parallel_processing
        
        results = {}
        
        if use_parallel and len(processors) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit tasks
                future_to_processor = {
                    executor.submit(self._process_single, name, data): name
                    for name in processors
                }
                
                # Collect results
                for future in as_completed(future_to_processor):
                    processor_name = future_to_processor[future]
                    try:
                        result = future.result()
                        results[processor_name] = result
                    except Exception as e:
                        self.logger.error(f"Error processing with {processor_name}: {str(e)}")
                        results[processor_name] = ProcessingResult(
                            data=None,
                            metadata={"error": str(e)},
                            timestamp=None,
                            processor_type=processor_name,
                            errors=[str(e)]
                        )
        else:
            # Sequential processing
            for processor_name in processors:
                try:
                    result = self._process_single(processor_name, data)
                    results[processor_name] = result
                except Exception as e:
                    self.logger.error(f"Error processing with {processor_name}: {str(e)}")
                    results[processor_name] = ProcessingResult(
                        data=None,
                        metadata={"error": str(e)},
                        timestamp=None,
                        processor_type=processor_name,
                        errors=[str(e)]
                    )
        
        # Save results if configured
        if self.config.save_intermediate_results:
            self._save_results(results)
        
        return results
    
    def _process_single(self, processor_name: str, data: Any) -> ProcessingResult:
        """Process data with a single processor."""
        processor = self.processors[processor_name]
        
        # Validate input
        if not processor.validate_input(data):
            raise ValueError(f"Invalid input data for processor {processor_name}")
        
        # Process data
        result = processor.process(data)
        
        self.logger.debug(f"Completed processing with {processor_name}")
        return result
    
    def _save_results(self, results: Dict[str, ProcessingResult]) -> None:
        """Save processing results to cache."""
        try:
            cache_path = Path(self.config.cache_directory)
            timestamp = results[list(results.keys())[0]].timestamp.strftime("%Y%m%d_%H%M%S")
            
            if self.config.output_format == "pickle":
                filename = cache_path / f"results_{timestamp}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(results, f)
            
            elif self.config.output_format == "json":
                filename = cache_path / f"results_{timestamp}.json"
                # Convert results to JSON-serializable format
                json_results = {}
                for name, result in results.items():
                    json_results[name] = {
                        "metadata": result.metadata,
                        "timestamp": result.timestamp.isoformat(),
                        "processor_type": result.processor_type,
                        "confidence": result.confidence,
                        "errors": result.errors,
                        # Note: data field may not be JSON serializable
                    }
                with open(filename, 'w') as f:
                    json.dump(json_results, f, indent=2)
            
            elif self.config.output_format == "hdf5":
                filename = cache_path / f"results_{timestamp}.h5"
                with h5py.File(filename, 'w') as f:
                    for name, result in results.items():
                        group = f.create_group(name)
                        group.attrs['processor_type'] = result.processor_type
                        group.attrs['timestamp'] = result.timestamp.isoformat()
                        if result.confidence is not None:
                            group.attrs['confidence'] = result.confidence
                        
                        # Save data if it's numpy array
                        if isinstance(result.data, np.ndarray):
                            group.create_dataset('data', data=result.data)
                        
                        # Save metadata
                        metadata_group = group.create_group('metadata')
                        for key, value in result.metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                metadata_group.attrs[key] = value
            
            self.logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save results: {str(e)}")
    
    def get_processor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all processors."""
        status = {}
        for name, processor in self.processors.items():
            status[name] = {
                "name": processor.name,
                "ready": processor.is_ready(),
                "config": processor.get_config()
            }
        return status
    
    def analyze_consciousness_landscape(self, 
                                      data: Any,
                                      generate_report: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive consciousness analysis across all domains.
        
        Args:
            data: Input data for analysis
            generate_report: Whether to generate summary report
            
        Returns:
            Comprehensive analysis results
        """
        # Process data through all available processors
        results = self.process_data(data)
        
        # Extract key metrics from each processor
        consciousness_metrics = {}
        
        for processor_name, result in results.items():
            if result.errors:
                consciousness_metrics[processor_name] = {
                    "status": "error",
                    "errors": result.errors
                }
                continue
            
            # Extract consciousness-relevant metrics
            if hasattr(result, 'data') and isinstance(result.data, dict):
                consciousness_metrics[processor_name] = {
                    "status": "success",
                    "confidence": result.confidence,
                    "key_metrics": result.data
                }
            else:
                consciousness_metrics[processor_name] = {
                    "status": "success",
                    "confidence": result.confidence,
                    "processed": True
                }
        
        # Generate integrated analysis
        integrated_analysis = self._integrate_consciousness_metrics(consciousness_metrics)
        
        analysis_result = {
            "individual_results": consciousness_metrics,
            "integrated_analysis": integrated_analysis,
            "timestamp": results[list(results.keys())[0]].timestamp.isoformat(),
            "framework_version": "0.1.0"
        }
        
        if generate_report:
            report = self._generate_consciousness_report(analysis_result)
            analysis_result["report"] = report
        
        return analysis_result
    
    def _integrate_consciousness_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate consciousness metrics across different modalities."""
        integration = {
            "overall_consciousness_score": 0.0,
            "domain_contributions": {},
            "cross_domain_correlations": {},
            "confidence_weighted_average": 0.0
        }
        
        # Calculate weighted consciousness score
        total_weight = 0
        weighted_score = 0
        
        for domain, result in metrics.items():
            if result.get("status") == "success":
                confidence = result.get("confidence", 0.5)
                # Extract consciousness score from key metrics
                if "key_metrics" in result:
                    domain_score = self._extract_consciousness_score(result["key_metrics"])
                else:
                    domain_score = 0.5  # Default neutral score
                
                integration["domain_contributions"][domain] = {
                    "score": domain_score,
                    "confidence": confidence,
                    "weight": confidence
                }
                
                weighted_score += domain_score * confidence
                total_weight += confidence
        
        if total_weight > 0:
            integration["confidence_weighted_average"] = weighted_score / total_weight
            integration["overall_consciousness_score"] = integration["confidence_weighted_average"]
        
        return integration
    
    def _extract_consciousness_score(self, metrics: Dict[str, Any]) -> float:
        """Extract a unified consciousness score from domain-specific metrics."""
        # Look for common consciousness indicators
        consciousness_indicators = [
            "phi", "complexity", "integration", "consciousness_probability",
            "pci_score", "connectivity_strength", "quantum_coherence"
        ]
        
        scores = []
        for indicator in consciousness_indicators:
            if indicator in metrics:
                value = metrics[indicator]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    # Normalize to 0-1 range (simple heuristic)
                    normalized_score = min(1.0, max(0.0, float(value)))
                    scores.append(normalized_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _generate_consciousness_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable consciousness analysis report."""
        report_lines = [
            "=== QUANTUM CONSCIOUSNESS ANALYSIS REPORT ===",
            f"Analysis Timestamp: {analysis['timestamp']}",
            f"Framework Version: {analysis['framework_version']}",
            "",
            "INTEGRATED CONSCIOUSNESS ASSESSMENT:",
            f"Overall Consciousness Score: {analysis['integrated_analysis']['overall_consciousness_score']:.3f}",
            f"Confidence-Weighted Average: {analysis['integrated_analysis']['confidence_weighted_average']:.3f}",
            "",
            "DOMAIN-SPECIFIC RESULTS:"
        ]
        
        for domain, result in analysis['individual_results'].items():
            report_lines.append(f"\n{domain.upper()}:")
            if result['status'] == 'success':
                report_lines.append(f"  Status: SUCCESS")
                if 'confidence' in result and result['confidence']:
                    report_lines.append(f"  Confidence: {result['confidence']:.3f}")
                if domain in analysis['integrated_analysis']['domain_contributions']:
                    score = analysis['integrated_analysis']['domain_contributions'][domain]['score']
                    report_lines.append(f"  Consciousness Score: {score:.3f}")
            else:
                report_lines.append(f"  Status: ERROR")
                if 'errors' in result:
                    for error in result['errors']:
                        report_lines.append(f"  Error: {error}")
        
        report_lines.extend([
            "",
            "=== END REPORT ===",
            ""
        ])
        
        return "\n".join(report_lines)