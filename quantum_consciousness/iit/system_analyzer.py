"""
System structure analyzer for IIT.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import networkx as nx


class SystemAnalyzer:
    """
    Analyzes system structure and topology for IIT calculations.
    """
    
    def __init__(self, max_system_size: int = 10):
        self.max_system_size = max_system_size
    
    def analyze_structure(self, connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Analyze structural properties of the system.
        
        Args:
            connectivity_matrix: System connectivity matrix
            
        Returns:
            Dictionary with structural metrics
        """
        if connectivity_matrix.size == 0:
            return {}
        
        # Create network graph
        G = self._create_network_graph(connectivity_matrix)
        
        metrics = {}
        
        # Basic network metrics
        metrics['density'] = nx.density(G)
        metrics['clustering'] = nx.average_clustering(G)
        
        # Connectivity metrics
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
            metrics['radius'] = nx.radius(G)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) > 1:
                metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['radius'] = nx.radius(subgraph)
            else:
                metrics['average_path_length'] = 0
                metrics['diameter'] = 0
                metrics['radius'] = 0
        
        # Small-world metrics
        metrics['small_world_coefficient'] = self._calculate_small_world_coefficient(G)
        
        # Modularity
        metrics['modularity'] = self._calculate_modularity(G)
        
        # Integration metrics
        metrics['integration'] = self._calculate_integration(connectivity_matrix)
        metrics['differentiation'] = self._calculate_differentiation(connectivity_matrix)
        
        # Efficiency metrics
        metrics['global_efficiency'] = self._calculate_global_efficiency(connectivity_matrix)
        metrics['local_efficiency'] = self._calculate_local_efficiency(connectivity_matrix)
        
        return metrics
    
    def identify_main_complex(self, connectivity_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Identify the main complex (Φ-complex) in the system.
        
        Args:
            connectivity_matrix: System connectivity matrix
            
        Returns:
            Dictionary describing the main complex
        """
        n_nodes = connectivity_matrix.shape[0]
        
        if n_nodes <= 1:
            return {"nodes": list(range(n_nodes)), "phi": 0.0, "size": n_nodes}
        
        # Find all possible subsystems and their Phi values
        best_complex = {"nodes": list(range(n_nodes)), "phi": 0.0, "size": n_nodes}
        
        # For computational tractability, limit to reasonable subsystem sizes
        max_subset_size = min(self.max_system_size, n_nodes)
        
        from itertools import combinations
        
        for size in range(2, max_subset_size + 1):
            for subset in combinations(range(n_nodes), size):
                subset_matrix = connectivity_matrix[np.ix_(subset, subset)]
                
                # Calculate Phi for this subset (simplified)
                phi_value = self._estimate_subset_phi(subset_matrix)
                
                if phi_value > best_complex["phi"]:
                    best_complex = {
                        "nodes": list(subset),
                        "phi": phi_value,
                        "size": len(subset)
                    }
        
        return best_complex
    
    def find_optimal_partitions(self, connectivity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find optimal partitions of the system.
        
        Args:
            connectivity_matrix: System connectivity matrix
            
        Returns:
            List of partition descriptions
        """
        n_nodes = connectivity_matrix.shape[0]
        
        if n_nodes <= 2:
            return [{"partition": [list(range(n_nodes))], "cut_strength": 0.0}]
        
        # Use spectral clustering to find natural partitions
        partitions = []
        
        # Try different numbers of clusters
        for n_clusters in range(2, min(n_nodes, 5) + 1):
            partition = self._spectral_partition(connectivity_matrix, n_clusters)
            cut_strength = self._calculate_cut_strength(connectivity_matrix, partition)
            
            partitions.append({
                "partition": partition,
                "n_clusters": n_clusters,
                "cut_strength": cut_strength
            })
        
        # Sort by cut strength (lower is better for finding natural divisions)
        partitions.sort(key=lambda x: x["cut_strength"])
        
        return partitions
    
    def _create_network_graph(self, connectivity_matrix: np.ndarray) -> nx.Graph:
        """Create NetworkX graph from connectivity matrix."""
        # Threshold small values to create cleaner graph
        threshold = np.mean(np.abs(connectivity_matrix)) * 0.1
        thresholded = np.where(np.abs(connectivity_matrix) > threshold, connectivity_matrix, 0)
        
        G = nx.from_numpy_array(thresholded)
        return G
    
    def _calculate_small_world_coefficient(self, G: nx.Graph) -> float:
        """Calculate small-world coefficient."""
        try:
            if len(G) < 4:
                return 0.0
            
            # Actual clustering and path length
            C = nx.average_clustering(G)
            
            if nx.is_connected(G):
                L = nx.average_shortest_path_length(G)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) > 1:
                    L = nx.average_shortest_path_length(G.subgraph(largest_cc))
                else:
                    return 0.0
            
            # Random network comparison
            n = len(G)
            m = G.number_of_edges()
            
            if m == 0:
                return 0.0
            
            # Expected values for random network
            p = 2 * m / (n * (n - 1))  # Connection probability
            C_rand = p  # Expected clustering coefficient
            L_rand = np.log(n) / np.log(2 * m / n) if m > 0 else float('inf')
            
            # Small-world coefficient
            if C_rand > 0 and L_rand > 0:
                sigma = (C / C_rand) / (L / L_rand)
                return sigma
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _calculate_modularity(self, G: nx.Graph) -> float:
        """Calculate network modularity using community detection."""
        try:
            if len(G) < 2 or G.number_of_edges() == 0:
                return 0.0
            
            # Use Louvain community detection
            import community  # python-louvain
            communities = community.best_partition(G)
            modularity = community.modularity(communities, G)
            return modularity
        except ImportError:
            # Fallback: simple modularity estimation
            return self._estimate_modularity(G)
        except:
            return 0.0
    
    def _estimate_modularity(self, G: nx.Graph) -> float:
        """Simple modularity estimation without external libraries."""
        if len(G) < 2:
            return 0.0
        
        # Use spectral clustering as proxy for communities
        adj_matrix = nx.adjacency_matrix(G).toarray()
        
        try:
            # Compute Laplacian
            degree = np.sum(adj_matrix, axis=1)
            L = np.diag(degree) - adj_matrix
            
            # Find second smallest eigenvalue (Fiedler value)
            eigenvals, eigenvecs = np.linalg.eigh(L)
            fiedler_vec = eigenvecs[:, 1]  # Second eigenvector
            
            # Simple binary partition based on sign of Fiedler vector
            community1 = np.where(fiedler_vec >= 0)[0]
            community2 = np.where(fiedler_vec < 0)[0]
            
            # Calculate modularity for this partition
            m = G.number_of_edges()
            if m == 0:
                return 0.0
            
            Q = 0
            for i in G.nodes():
                for j in G.nodes():
                    if (i in community1 and j in community1) or (i in community2 and j in community2):
                        A_ij = 1 if G.has_edge(i, j) else 0
                        k_i = G.degree(i)
                        k_j = G.degree(j)
                        Q += A_ij - (k_i * k_j) / (2 * m)
            
            return Q / (2 * m)
        except:
            return 0.0
    
    def _calculate_integration(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate integration measure."""
        return self._calculate_global_efficiency(connectivity_matrix)
    
    def _calculate_differentiation(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate differentiation measure."""
        # Use variance of connectivity strengths as differentiation proxy
        strengths = np.sum(np.abs(connectivity_matrix), axis=1)
        if len(strengths) <= 1:
            return 0.0
        
        differentiation = np.std(strengths) / (np.mean(strengths) + 1e-10)
        return min(1.0, differentiation)
    
    def _calculate_global_efficiency(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate global efficiency."""
        n = connectivity_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # Convert to distance matrix (inverse of connectivity)
        with np.errstate(divide='ignore', invalid='ignore'):
            distance_matrix = 1.0 / (np.abs(connectivity_matrix) + 1e-10)
        
        # Set diagonal to 0
        np.fill_diagonal(distance_matrix, 0)
        
        # Calculate all-pairs shortest paths using Floyd-Warshall
        distances = distance_matrix.copy()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
        
        # Calculate global efficiency
        efficiency = 0.0
        count = 0
        
        for i in range(n):
            for j in range(n):
                if i != j and distances[i, j] != np.inf and distances[i, j] > 0:
                    efficiency += 1.0 / distances[i, j]
                    count += 1
        
        return efficiency / count if count > 0 else 0.0
    
    def _calculate_local_efficiency(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate local efficiency."""
        n = connectivity_matrix.shape[0]
        if n <= 2:
            return 0.0
        
        local_efficiencies = []
        
        for i in range(n):
            # Find neighbors of node i
            neighbors = np.where(np.abs(connectivity_matrix[i, :]) > 1e-10)[0]
            neighbors = neighbors[neighbors != i]  # Remove self
            
            if len(neighbors) <= 1:
                local_efficiencies.append(0.0)
                continue
            
            # Create subgraph of neighbors
            subgraph_matrix = connectivity_matrix[np.ix_(neighbors, neighbors)]
            
            # Calculate efficiency of subgraph
            subgraph_efficiency = self._calculate_global_efficiency(subgraph_matrix)
            local_efficiencies.append(subgraph_efficiency)
        
        return np.mean(local_efficiencies)
    
    def _spectral_partition(self, connectivity_matrix: np.ndarray, n_clusters: int) -> List[List[int]]:
        """Partition nodes using spectral clustering."""
        try:
            from sklearn.cluster import SpectralClustering
            
            # Use absolute values for clustering
            abs_matrix = np.abs(connectivity_matrix)
            
            clustering = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed',
                random_state=42
            )
            
            labels = clustering.fit_predict(abs_matrix)
            
            # Group nodes by cluster labels
            partitions = []
            for cluster_id in range(n_clusters):
                cluster_nodes = np.where(labels == cluster_id)[0].tolist()
                if cluster_nodes:  # Only add non-empty clusters
                    partitions.append(cluster_nodes)
            
            return partitions
            
        except ImportError:
            # Fallback: simple binary partition using Fiedler vector
            return self._fiedler_partition(connectivity_matrix)
    
    def _fiedler_partition(self, connectivity_matrix: np.ndarray) -> List[List[int]]:
        """Simple binary partition using Fiedler vector."""
        n = connectivity_matrix.shape[0]
        
        try:
            # Compute graph Laplacian
            degree = np.sum(np.abs(connectivity_matrix), axis=1)
            L = np.diag(degree) - np.abs(connectivity_matrix)
            
            # Find Fiedler vector (eigenvector of second smallest eigenvalue)
            eigenvals, eigenvecs = np.linalg.eigh(L)
            fiedler_vec = eigenvecs[:, 1]
            
            # Partition based on sign
            partition1 = np.where(fiedler_vec >= 0)[0].tolist()
            partition2 = np.where(fiedler_vec < 0)[0].tolist()
            
            return [partition1, partition2] if partition1 and partition2 else [[*range(n)]]
            
        except:
            # Ultimate fallback: simple split
            mid = n // 2
            return [list(range(mid)), list(range(mid, n))]
    
    def _calculate_cut_strength(self, connectivity_matrix: np.ndarray, partitions: List[List[int]]) -> float:
        """Calculate strength of cuts between partitions."""
        if len(partitions) <= 1:
            return 0.0
        
        total_cut_strength = 0.0
        
        for i, part1 in enumerate(partitions):
            for j, part2 in enumerate(partitions):
                if i < j:  # Avoid double counting
                    # Calculate connection strength between partitions
                    cut_strength = 0.0
                    for node1 in part1:
                        for node2 in part2:
                            cut_strength += abs(connectivity_matrix[node1, node2])
                    
                    total_cut_strength += cut_strength
        
        return total_cut_strength
    
    def _estimate_subset_phi(self, subset_matrix: np.ndarray) -> float:
        """Estimate Phi for a subset (simplified calculation)."""
        if subset_matrix.shape[0] <= 1:
            return 0.0
        
        # Use network efficiency as Phi proxy
        efficiency = self._calculate_global_efficiency(subset_matrix)
        
        # Normalize by system size
        size_factor = min(1.0, subset_matrix.shape[0] / 5.0)
        
        return efficiency * size_factor