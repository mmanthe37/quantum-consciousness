"""
Attention mechanism analyzer for consciousness detection.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import entropy


class AttentionAnalyzer:
    """
    Analyzer for attention mechanisms in neural networks.
    
    Analyzes attention patterns for consciousness indicators including:
    - Global attention coherence
    - Attention integration across heads/layers
    - Temporal attention dynamics
    - Broadcasting patterns
    """
    
    def __init__(self, temporal_window: int = 10):
        self.temporal_window = temporal_window
    
    def analyze_attention_integration(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """
        Analyze attention integration patterns.
        
        Args:
            attention_weights: Attention weights (heads x queries x keys) or (queries x keys)
            
        Returns:
            Dictionary with integration metrics
        """
        results = {}
        
        if len(attention_weights.shape) == 2:
            # Single attention matrix
            attention_weights = attention_weights[np.newaxis, :, :]
        
        n_heads, n_queries, n_keys = attention_weights.shape
        
        # Multi-head integration
        if n_heads > 1:
            results['multi_head_integration'] = self._calculate_multi_head_integration(attention_weights)
            results['head_diversity'] = self._calculate_head_diversity(attention_weights)
            results['head_coherence'] = self._calculate_head_coherence(attention_weights)
        
        # Query-key integration
        results['query_key_integration'] = self._calculate_query_key_integration(attention_weights)
        
        # Global attention patterns
        results['global_attention_strength'] = self._calculate_global_attention_strength(attention_weights)
        
        return results
    
    def calculate_attention_coherence(self, attention_weights: np.ndarray) -> float:
        """
        Calculate overall attention coherence.
        
        Args:
            attention_weights: Attention weights
            
        Returns:
            Coherence score (0-1)
        """
        if attention_weights.size == 0:
            return 0.0
        
        # Ensure 3D shape
        if len(attention_weights.shape) == 2:
            attention_weights = attention_weights[np.newaxis, :, :]
        
        coherence_scores = []
        
        # Calculate coherence for each attention head
        for head_idx in range(attention_weights.shape[0]):
            head_attention = attention_weights[head_idx]
            
            # Coherence as consistency of attention patterns
            coherence = self._calculate_single_head_coherence(head_attention)
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def calculate_attention_complexity(self, attention_weights: np.ndarray) -> float:
        """
        Calculate attention complexity.
        
        Args:
            attention_weights: Attention weights
            
        Returns:
            Complexity score
        """
        if attention_weights.size == 0:
            return 0.0
        
        # Ensure 3D shape
        if len(attention_weights.shape) == 2:
            attention_weights = attention_weights[np.newaxis, :, :]
        
        complexity_scores = []
        
        for head_idx in range(attention_weights.shape[0]):
            head_attention = attention_weights[head_idx]
            
            # Calculate entropy-based complexity
            head_complexity = self._calculate_attention_entropy(head_attention)
            complexity_scores.append(head_complexity)
        
        return np.mean(complexity_scores) if complexity_scores else 0.0
    
    def analyze_temporal_attention(self, 
                                 temporal_attention: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze temporal attention dynamics.
        
        Args:
            temporal_attention: List of attention weights over time
            
        Returns:
            Dictionary with temporal metrics
        """
        results = {}
        
        if len(temporal_attention) < 2:
            return results
        
        # Calculate temporal consistency
        results['temporal_consistency'] = self._calculate_temporal_consistency(temporal_attention)
        
        # Calculate attention persistence
        results['attention_persistence'] = self._calculate_attention_persistence(temporal_attention)
        
        # Calculate attention dynamics
        results['attention_dynamics'] = self._calculate_attention_dynamics(temporal_attention)
        
        return results
    
    def detect_global_workspace_patterns(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """
        Detect global workspace patterns in attention.
        
        Args:
            attention_weights: Attention weights
            
        Returns:
            Dictionary with global workspace metrics
        """
        results = {}
        
        if attention_weights.size == 0:
            return results
        
        # Ensure 3D shape
        if len(attention_weights.shape) == 2:
            attention_weights = attention_weights[np.newaxis, :, :]
        
        # Broadcasting strength
        results['broadcasting_strength'] = self._calculate_broadcasting_strength(attention_weights)
        
        # Global access patterns
        results['global_access'] = self._calculate_global_access(attention_weights)
        
        # Competition dynamics
        results['attention_competition'] = self._calculate_attention_competition(attention_weights)
        
        return results
    
    def _calculate_multi_head_integration(self, attention_weights: np.ndarray) -> float:
        """Calculate integration across attention heads."""
        n_heads = attention_weights.shape[0]
        
        if n_heads <= 1:
            return 0.0
        
        # Calculate pairwise correlations between heads
        correlations = []
        
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                head_i = attention_weights[i].flatten()
                head_j = attention_weights[j].flatten()
                
                # Calculate correlation
                if np.std(head_i) > 1e-10 and np.std(head_j) > 1e-10:
                    corr = np.corrcoef(head_i, head_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_head_diversity(self, attention_weights: np.ndarray) -> float:
        """Calculate diversity across attention heads."""
        n_heads = attention_weights.shape[0]
        
        if n_heads <= 1:
            return 0.0
        
        # Calculate pairwise distances between heads
        distances = []
        
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                head_i = attention_weights[i]
                head_j = attention_weights[j]
                
                # Calculate Jensen-Shannon divergence
                js_div = self._jensen_shannon_divergence(head_i, head_j)
                distances.append(js_div)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_head_coherence(self, attention_weights: np.ndarray) -> float:
        """Calculate coherence across attention heads."""
        n_heads = attention_weights.shape[0]
        
        if n_heads <= 1:
            return 1.0
        
        # Calculate how coherently heads attend to the same positions
        mean_attention = np.mean(attention_weights, axis=0)
        
        coherence_scores = []
        for head_idx in range(n_heads):
            head_attention = attention_weights[head_idx]
            
            # Correlation with mean attention pattern
            flat_head = head_attention.flatten()
            flat_mean = mean_attention.flatten()
            
            if np.std(flat_head) > 1e-10 and np.std(flat_mean) > 1e-10:
                corr = np.corrcoef(flat_head, flat_mean)[0, 1]
                if not np.isnan(corr):
                    coherence_scores.append(abs(corr))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_query_key_integration(self, attention_weights: np.ndarray) -> float:
        """Calculate integration between queries and keys."""
        # Average attention weights to get overall query-key interaction strength
        mean_attention = np.mean(attention_weights, axis=0)
        
        # Calculate effective integration as entropy of attention distribution
        attention_entropy = self._calculate_attention_entropy(mean_attention)
        
        # Normalize entropy by maximum possible entropy
        n_keys = mean_attention.shape[1]
        max_entropy = np.log2(n_keys) if n_keys > 1 else 1.0
        
        normalized_entropy = attention_entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _calculate_global_attention_strength(self, attention_weights: np.ndarray) -> float:
        """Calculate strength of global attention patterns."""
        # Global attention as variance in attention distribution
        mean_attention = np.mean(attention_weights, axis=0)
        
        # Calculate coefficient of variation for each query
        cv_scores = []
        for query_idx in range(mean_attention.shape[0]):
            query_attention = mean_attention[query_idx]
            
            if np.mean(query_attention) > 1e-10:
                cv = np.std(query_attention) / np.mean(query_attention)
                cv_scores.append(cv)
        
        return np.mean(cv_scores) if cv_scores else 0.0
    
    def _calculate_single_head_coherence(self, attention_matrix: np.ndarray) -> float:
        """Calculate coherence for a single attention head."""
        # Coherence as consistency of attention patterns across queries
        query_similarities = []
        
        n_queries = attention_matrix.shape[0]
        
        for i in range(min(10, n_queries)):  # Limit for efficiency
            for j in range(i + 1, min(10, n_queries)):
                query_i = attention_matrix[i]
                query_j = attention_matrix[j]
                
                # Calculate cosine similarity
                dot_product = np.dot(query_i, query_j)
                norm_i = np.linalg.norm(query_i)
                norm_j = np.linalg.norm(query_j)
                
                if norm_i > 1e-10 and norm_j > 1e-10:
                    similarity = dot_product / (norm_i * norm_j)
                    query_similarities.append(abs(similarity))
        
        return np.mean(query_similarities) if query_similarities else 0.0
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        if attention_matrix.size == 0:
            return 0.0
        
        # Flatten and normalize
        flat_attention = attention_matrix.flatten()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        flat_attention = flat_attention + epsilon
        flat_attention = flat_attention / np.sum(flat_attention)
        
        # Calculate entropy
        attention_entropy = entropy(flat_attention, base=2)
        
        return attention_entropy
    
    def _calculate_temporal_consistency(self, temporal_attention: List[np.ndarray]) -> float:
        """Calculate consistency of attention patterns over time."""
        if len(temporal_attention) < 2:
            return 0.0
        
        # Calculate pairwise correlations between consecutive time steps
        correlations = []
        
        for t in range(len(temporal_attention) - 1):
            att_t = temporal_attention[t].flatten()
            att_t_plus_1 = temporal_attention[t + 1].flatten()
            
            if np.std(att_t) > 1e-10 and np.std(att_t_plus_1) > 1e-10:
                corr = np.corrcoef(att_t, att_t_plus_1)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_attention_persistence(self, temporal_attention: List[np.ndarray]) -> float:
        """Calculate persistence of attention patterns."""
        if len(temporal_attention) < self.temporal_window:
            return 0.0
        
        # Calculate how long attention patterns persist
        persistence_scores = []
        
        for start_t in range(len(temporal_attention) - self.temporal_window + 1):
            window_attention = temporal_attention[start_t:start_t + self.temporal_window]
            
            # Calculate stability within window
            stability = self._calculate_window_stability(window_attention)
            persistence_scores.append(stability)
        
        return np.mean(persistence_scores) if persistence_scores else 0.0
    
    def _calculate_attention_dynamics(self, temporal_attention: List[np.ndarray]) -> float:
        """Calculate dynamics of attention changes."""
        if len(temporal_attention) < 2:
            return 0.0
        
        # Calculate rate of change in attention patterns
        change_rates = []
        
        for t in range(len(temporal_attention) - 1):
            att_t = temporal_attention[t]
            att_t_plus_1 = temporal_attention[t + 1]
            
            # Calculate change magnitude
            change = np.linalg.norm(att_t_plus_1 - att_t)
            change_rates.append(change)
        
        # Dynamics as coefficient of variation of change rates
        if len(change_rates) > 1 and np.mean(change_rates) > 1e-10:
            dynamics = np.std(change_rates) / np.mean(change_rates)
        else:
            dynamics = 0.0
        
        return dynamics
    
    def _calculate_broadcasting_strength(self, attention_weights: np.ndarray) -> float:
        """Calculate strength of attention broadcasting."""
        # Broadcasting as tendency for queries to attend to same keys
        mean_attention = np.mean(attention_weights, axis=0)
        
        # Calculate overlap in attention patterns
        overlap_scores = []
        
        for query_i in range(min(10, mean_attention.shape[0])):
            for query_j in range(query_i + 1, min(10, mean_attention.shape[0])):
                att_i = mean_attention[query_i]
                att_j = mean_attention[query_j]
                
                # Calculate overlap (dot product of normalized vectors)
                norm_i = np.linalg.norm(att_i)
                norm_j = np.linalg.norm(att_j)
                
                if norm_i > 1e-10 and norm_j > 1e-10:
                    overlap = np.dot(att_i, att_j) / (norm_i * norm_j)
                    overlap_scores.append(abs(overlap))
        
        return np.mean(overlap_scores) if overlap_scores else 0.0
    
    def _calculate_global_access(self, attention_weights: np.ndarray) -> float:
        """Calculate global accessibility through attention."""
        # Global access as entropy of aggregated attention
        aggregated_attention = np.sum(attention_weights, axis=(0, 1))  # Sum over heads and queries
        
        # Normalize
        if np.sum(aggregated_attention) > 0:
            aggregated_attention = aggregated_attention / np.sum(aggregated_attention)
        
        # Calculate entropy
        global_entropy = entropy(aggregated_attention + 1e-10, base=2)
        
        # Normalize by maximum entropy
        max_entropy = np.log2(len(aggregated_attention))
        
        return global_entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_attention_competition(self, attention_weights: np.ndarray) -> float:
        """Calculate competition dynamics in attention."""
        # Competition as variance in attention distributions
        competition_scores = []
        
        for head_idx in range(attention_weights.shape[0]):
            head_attention = attention_weights[head_idx]
            
            # Calculate variance across keys for each query
            query_variances = np.var(head_attention, axis=1)
            competition_scores.extend(query_variances)
        
        return np.mean(competition_scores) if competition_scores else 0.0
    
    def _jensen_shannon_divergence(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        # Normalize distributions
        dist1_flat = dist1.flatten()
        dist2_flat = dist2.flatten()
        
        # Ensure same length
        min_len = min(len(dist1_flat), len(dist2_flat))
        dist1_flat = dist1_flat[:min_len]
        dist2_flat = dist2_flat[:min_len]
        
        # Add epsilon and normalize
        epsilon = 1e-10
        dist1_norm = (dist1_flat + epsilon) / np.sum(dist1_flat + epsilon)
        dist2_norm = (dist2_flat + epsilon) / np.sum(dist2_flat + epsilon)
        
        # Calculate JS divergence
        m = 0.5 * (dist1_norm + dist2_norm)
        
        js_div = 0.5 * entropy(dist1_norm, m, base=2) + 0.5 * entropy(dist2_norm, m, base=2)
        
        return js_div
    
    def _calculate_window_stability(self, window_attention: List[np.ndarray]) -> float:
        """Calculate stability within a temporal window."""
        if len(window_attention) < 2:
            return 1.0
        
        # Calculate pairwise correlations within window
        correlations = []
        
        for i in range(len(window_attention)):
            for j in range(i + 1, len(window_attention)):
                att_i = window_attention[i].flatten()
                att_j = window_attention[j].flatten()
                
                if np.std(att_i) > 1e-10 and np.std(att_j) > 1e-10:
                    corr = np.corrcoef(att_i, att_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0