"""
Connectivity analysis for consciousness assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import hilbert
from scipy.stats import pearsonr
import networkx as nx


class ConnectivityAnalyzer:
    """
    Connectivity analyzer for consciousness indicators.
    
    Analyzes connectivity patterns for:
    - Static and dynamic functional connectivity
    - Network topology and graph metrics
    - Community structure
    - Metastability and flexibility
    """
    
    def __init__(self, method: str = 'pearson'):
        self.method = method
    
    def analyze_network_topology(self, connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze network topology metrics."""
        results = {}
        
        if connectivity_matrix.size == 0:
            return results
        
        # Threshold connectivity matrix to create binary network
        threshold = np.percentile(np.abs(connectivity_matrix), 75)  # Top 25% connections
        binary_matrix = (np.abs(connectivity_matrix) > threshold).astype(int)
        np.fill_diagonal(binary_matrix, 0)
        
        # Create network graph
        G = nx.from_numpy_array(binary_matrix)
        
        if len(G) > 0:
            # Basic metrics
            results['network_density'] = nx.density(G)
            results['average_clustering'] = nx.average_clustering(G)
            
            # Path-based metrics
            if nx.is_connected(G):
                results['average_path_length'] = nx.average_shortest_path_length(G)
                results['diameter'] = nx.diameter(G)
                results['radius'] = nx.radius(G)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len, default=[])
                if len(largest_cc) > 1:
                    subgraph = G.subgraph(largest_cc)
                    results['average_path_length'] = nx.average_shortest_path_length(subgraph)
                    results['diameter'] = nx.diameter(subgraph)
                    results['radius'] = nx.radius(subgraph)
                else:
                    results['average_path_length'] = 0
                    results['diameter'] = 0
                    results['radius'] = 0
        
        return results
    
    def calculate_graph_metrics(self, connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate graph theory metrics."""
        results = {}
        
        if connectivity_matrix.size == 0:
            return results
        
        # Weighted network analysis
        weighted_matrix = np.abs(connectivity_matrix)
        np.fill_diagonal(weighted_matrix, 0)
        
        G = nx.from_numpy_array(weighted_matrix)
        
        if len(G) > 0:
            # Efficiency metrics
            results['global_efficiency'] = nx.global_efficiency(G)
            results['local_efficiency'] = nx.local_efficiency(G)
            
            # Centrality measures (average across nodes)
            betweenness = nx.betweenness_centrality(G, weight='weight')
            results['average_betweenness_centrality'] = np.mean(list(betweenness.values()))
            
            closeness = nx.closeness_centrality(G, distance='weight')
            results['average_closeness_centrality'] = np.mean(list(closeness.values()))
            
            eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            results['average_eigenvector_centrality'] = np.mean(list(eigenvector.values()))
            
            # Rich club coefficient (simplified)
            degrees = dict(G.degree(weight='weight'))
            if degrees:
                max_degree = max(degrees.values())
                high_degree_threshold = max_degree * 0.8  # Top 20% of nodes
                high_degree_nodes = [node for node, degree in degrees.items() if degree >= high_degree_threshold]
                
                if len(high_degree_nodes) > 1:
                    rich_club_subgraph = G.subgraph(high_degree_nodes)
                    rich_club_edges = rich_club_subgraph.number_of_edges()
                    max_possible_edges = len(high_degree_nodes) * (len(high_degree_nodes) - 1) / 2
                    
                    if max_possible_edges > 0:
                        results['rich_club_coefficient'] = rich_club_edges / max_possible_edges
        
        return results
    
    def detect_community_structure(self, connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Detect community structure in connectivity."""
        results = {}
        
        if connectivity_matrix.size == 0:
            return results
        
        # Create weighted network
        weighted_matrix = np.abs(connectivity_matrix)
        np.fill_diagonal(weighted_matrix, 0)
        
        G = nx.from_numpy_array(weighted_matrix)
        
        if len(G) > 1:
            try:
                # Modularity using Louvain algorithm (if available)
                import community as community_louvain
                communities = community_louvain.best_partition(G, weight='weight')
                modularity = community_louvain.modularity(communities, G, weight='weight')
                results['modularity'] = modularity
                
                # Number of communities
                results['number_of_communities'] = len(set(communities.values()))
                
            except ImportError:
                # Fallback: spectral modularity
                results.update(self._calculate_spectral_modularity(connectivity_matrix))
        
        return results
    
    def calculate_dynamic_connectivity(self, time_series: np.ndarray, window_size: int = 50, step_size: int = 10) -> np.ndarray:
        """Calculate dynamic functional connectivity."""
        if time_series.size == 0 or len(time_series.shape) < 2:
            return np.array([])
        
        n_regions, n_timepoints = time_series.shape
        
        if n_timepoints < window_size:
            # Static connectivity if insufficient time points
            return np.corrcoef(time_series)[np.newaxis, :, :]
        
        # Sliding window connectivity
        dynamic_conn_matrices = []
        
        for start_t in range(0, n_timepoints - window_size + 1, step_size):
            end_t = start_t + window_size
            window_data = time_series[:, start_t:end_t]
            
            # Calculate connectivity for this window
            if self.method == 'pearson':
                conn_matrix = np.corrcoef(window_data)
            elif self.method == 'mutual_information':
                conn_matrix = self._calculate_mutual_information_matrix(window_data)
            else:
                conn_matrix = np.corrcoef(window_data)
            
            dynamic_conn_matrices.append(conn_matrix)
        
        return np.array(dynamic_conn_matrices)
    
    def calculate_network_flexibility(self, dynamic_connectivity: np.ndarray) -> Dict[str, float]:
        """Calculate network flexibility metrics."""
        results = {}
        
        if dynamic_connectivity.size == 0:
            return results
        
        n_windows, n_regions, _ = dynamic_connectivity.shape
        
        if n_windows < 2:
            return results
        
        # Flexibility as variance in connectivity patterns
        connectivity_variance = np.var(dynamic_connectivity, axis=0)
        results['connectivity_flexibility'] = np.mean(connectivity_variance)
        
        # Node flexibility (how much each node's connectivity pattern changes)
        node_flexibilities = []
        
        for region in range(n_regions):
            region_connectivity_over_time = dynamic_connectivity[:, region, :]
            region_flexibility = np.var(region_connectivity_over_time, axis=0)
            node_flexibilities.append(np.mean(region_flexibility))
        
        results['average_node_flexibility'] = np.mean(node_flexibilities)
        results['node_flexibility_variance'] = np.var(node_flexibilities)
        
        return results
    
    def calculate_metastability(self, time_series: np.ndarray) -> float:
        """Calculate network metastability."""
        if time_series.size == 0 or len(time_series.shape) < 2:
            return 0.0
        
        n_regions, n_timepoints = time_series.shape
        
        if n_regions < 2 or n_timepoints < 10:
            return 0.0
        
        # Calculate instantaneous synchronization
        synchronization_over_time = []
        
        # Use sliding window to calculate instantaneous synchronization
        window_size = min(20, n_timepoints // 4)
        
        for t in range(0, n_timepoints - window_size + 1, window_size // 2):
            window_data = time_series[:, t:t + window_size]
            
            # Calculate pairwise correlations in this window
            correlations = []
            for i in range(n_regions):
                for j in range(i + 1, n_regions):
                    corr, _ = pearsonr(window_data[i], window_data[j])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                # Synchronization as mean correlation
                sync_value = np.mean(correlations)
                synchronization_over_time.append(sync_value)
        
        if len(synchronization_over_time) < 2:
            return 0.0
        
        # Metastability as standard deviation of synchronization over time
        metastability = np.std(synchronization_over_time)
        
        return float(metastability)
    
    def _calculate_spectral_modularity(self, connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate modularity using spectral methods."""
        results = {}
        
        try:
            # Create modularity matrix
            degrees = np.sum(np.abs(connectivity_matrix), axis=1)
            total_strength = np.sum(degrees)
            
            if total_strength == 0:
                results['modularity'] = 0.0
                return results
            
            # Modularity matrix
            expected_connectivity = np.outer(degrees, degrees) / total_strength
            modularity_matrix = connectivity_matrix - expected_connectivity
            
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(modularity_matrix)
            
            # Use leading eigenvector for binary partitioning
            leading_eigenvec = eigenvecs[:, np.argmax(eigenvals)]
            
            # Binary partition based on sign of leading eigenvector
            partition = (leading_eigenvec > 0).astype(int)
            
            # Calculate modularity of this partition
            modularity = 0.0
            n_nodes = len(partition)
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if partition[i] == partition[j]:  # Same community
                        expected_ij = degrees[i] * degrees[j] / total_strength
                        modularity += connectivity_matrix[i, j] - expected_ij
            
            modularity /= total_strength
            results['modularity'] = modularity
            
            # Number of communities (always 2 for binary partition)
            results['number_of_communities'] = 2
            
        except:
            results['modularity'] = 0.0
            results['number_of_communities'] = 1
        
        return results
    
    def _calculate_mutual_information_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate mutual information connectivity matrix."""
        n_regions = data.shape[0]
        mi_matrix = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(i, n_regions):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    mi_value = self._mutual_information(data[i], data[j])
                    mi_matrix[i, j] = mi_value
                    mi_matrix[j, i] = mi_value
        
        return mi_matrix
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Calculate mutual information between two signals."""
        try:
            # Discretize signals
            x_discrete = np.digitize(x, np.histogram(x, bins=bins)[1][:-1])
            y_discrete = np.digitize(y, np.histogram(y, bins=bins)[1][:-1])
            
            # Joint histogram
            joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=bins)
            joint_hist = joint_hist + 1e-10  # Avoid log(0)
            joint_prob = joint_hist / np.sum(joint_hist)
            
            # Marginal histograms
            x_hist = np.sum(joint_prob, axis=1)
            y_hist = np.sum(joint_prob, axis=0)
            
            # Mutual information
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if joint_prob[i, j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_hist[i] * y_hist[j]))
            
            return mi
        except:
            return 0.0