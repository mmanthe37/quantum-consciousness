"""
Base classes and interfaces for the quantum consciousness framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datetime import datetime


@dataclass
class ProcessingResult:
    """Standard result container for all processing operations."""
    
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    processor_type: str
    confidence: Optional[float] = None
    errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseProcessor(ABC):
    """Base class for all consciousness analysis processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the processor with required resources."""
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """Process input data and return structured results."""
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Validate input data format and requirements."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get current processor configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update processor configuration."""
        self.config.update(new_config)
    
    def is_ready(self) -> bool:
        """Check if processor is ready for operation."""
        return self._is_initialized


class DataValidator:
    """Utility class for common data validation operations."""
    
    @staticmethod
    def validate_neural_timeseries(data: np.ndarray) -> bool:
        """Validate neural time series data format."""
        if not isinstance(data, np.ndarray):
            return False
        if len(data.shape) < 2:
            return False
        if data.shape[0] < 1 or data.shape[1] < 1:
            return False
        if np.isnan(data).any() or np.isinf(data).any():
            return False
        return True
    
    @staticmethod
    def validate_network_structure(adjacency_matrix: np.ndarray) -> bool:
        """Validate network adjacency matrix."""
        if not isinstance(adjacency_matrix, np.ndarray):
            return False
        if len(adjacency_matrix.shape) != 2:
            return False
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            return False
        return True
    
    @staticmethod
    def validate_consciousness_features(features: Dict[str, float]) -> bool:
        """Validate consciousness feature dictionary."""
        if not isinstance(features, dict):
            return False
        if not features:
            return False
        for key, value in features.items():
            if not isinstance(value, (int, float, np.number)):
                return False
            if np.isnan(value) or np.isinf(value):
                return False
        return True


class ConsciousnessMetrics:
    """Common consciousness measurement metrics."""
    
    @staticmethod
    def phi_measure(system_state: np.ndarray, perturbation: np.ndarray) -> float:
        """Calculate integrated information (Phi) measure."""
        # Simplified Phi calculation - actual IIT implementation more complex
        if system_state.size == 0 or perturbation.size == 0:
            return 0.0
        
        # Calculate mutual information between system parts
        entropy_whole = -np.sum(system_state * np.log2(system_state + 1e-10))
        entropy_parts = 0.0
        
        # Split system into parts and calculate individual entropies
        n_parts = min(2, len(system_state) // 2)
        if n_parts > 1:
            part_size = len(system_state) // n_parts
            for i in range(n_parts):
                start_idx = i * part_size
                end_idx = start_idx + part_size if i < n_parts - 1 else len(system_state)
                part = system_state[start_idx:end_idx]
                if len(part) > 0:
                    part_norm = part / (np.sum(part) + 1e-10)
                    entropy_parts += -np.sum(part_norm * np.log2(part_norm + 1e-10))
        
        phi = max(0, entropy_parts - entropy_whole)
        return phi
    
    @staticmethod
    def complexity_measure(data: np.ndarray) -> float:
        """Calculate neural complexity measure."""
        if data.size == 0:
            return 0.0
        
        # Lempel-Ziv complexity approximation
        binary_data = (data > np.median(data)).astype(int)
        complexity = 0
        i = 0
        while i < len(binary_data):
            substring = ""
            j = i
            while j < len(binary_data):
                substring += str(binary_data[j])
                if substring not in str(binary_data[:j]):
                    complexity += 1
                    break
                j += 1
            i = j + 1 if j < len(binary_data) else len(binary_data)
        
        return complexity / len(binary_data) if len(binary_data) > 0 else 0.0
    
    @staticmethod
    def integration_measure(connectivity_matrix: np.ndarray) -> float:
        """Calculate network integration measure."""
        if connectivity_matrix.size == 0:
            return 0.0
        
        # Global efficiency as integration proxy
        n = connectivity_matrix.shape[0]
        if n < 2:
            return 0.0
        
        # Calculate shortest path lengths
        distances = np.full((n, n), np.inf)
        np.fill_diagonal(distances, 0)
        
        # Floyd-Warshall algorithm for shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
        
        # Global efficiency
        efficiency = 0.0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j and distances[i, j] != np.inf and distances[i, j] > 0:
                    efficiency += 1.0 / distances[i, j]
                    count += 1
        
        return efficiency / count if count > 0 else 0.0