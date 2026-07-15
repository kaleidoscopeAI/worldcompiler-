import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs
from dataclasses import dataclass, field
import asyncio
import math
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from sklearn.cluster import DBSCAN
from collections import defaultdict
import random
import threading
import os

class QuantumState:
    """Quantum state representation optimized for sparse computation"""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Use sparse representation for better memory efficiency
        self.amplitudes = {0: complex(1.0, 0.0)}  # Initialize to |0...0⟩
        
    def apply_hadamard(self, target: int):
        """Apply Hadamard gate to target qubit using sparse representation"""
        new_amplitudes = {}
        norm_factor = 1.0 / np.sqrt(2.0)
        
        for idx, amp in self.amplitudes.items():
            bit_val = (idx >> target) & 1
            paired_idx = idx ^ (1 << target)
            
            if bit_val == 0:
                new_amplitudes[idx] = new_amplitudes.get(idx, 0) + amp * norm_factor
                new_amplitudes[paired_idx] = new_amplitudes.get(paired_idx, 0) + amp * norm_factor
            else:
                new_amplitudes[idx] = new_amplitudes.get(idx, 0) + amp * norm_factor
                new_amplitudes[paired_idx] = new_amplitudes.get(paired_idx, 0) - amp * norm_factor
        
        # Remove very small amplitudes to maintain sparsity
        self.amplitudes = {k: v for k, v in new_amplitudes.items() if abs(v) > 1e-10}
    
    def apply_phase(self, target: int, theta: float):
        """Apply phase rotation to target qubit"""
        phase = complex(math.cos(theta), math.sin(theta))
        new_amplitudes = {}
        
        for idx, amp in self.amplitudes.items():
            if (idx >> target) & 1:
                new_amplitudes[idx] = amp * phase
            else:
                new_amplitudes[idx] = amp
        
        self.amplitudes = new_amplitudes
    
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate between control and target qubits"""
        new_amplitudes = {}
        
        for idx, amp in self.amplitudes.items():
            control_bit = (idx >> control) & 1
            if control_bit:
                flipped = idx ^ (1 << target)
                new_amplitudes[flipped] = amp
            else:
                new_amplitudes[idx] = amp
        
        self.amplitudes = new_amplitudes
    
    def apply_string_tension(self, tension: float):
        """Apply string tension to amplitudes based on Hamming weight"""
        new_amplitudes = {}
        norm_factor = 0.0
        
        for idx, amp in self.amplitudes.items():
            hamming = bin(idx).count('1')
            scale = 1.0 + (hamming / self.num_qubits - 0.5) * tension
            new_amp = amp * scale
            new_amplitudes[idx] = new_amp
            norm_factor += abs(new_amp)**2
        
        # Renormalize
        norm_factor = math.sqrt(norm_factor)
        if norm_factor > 0:
            self.amplitudes = {k: v / norm_factor for k, v in new_amplitudes.items()}
    
    def get_entropy(self) -> float:
        """Calculate von Neumann entropy (simplified)"""
        entropy = 0.0
        for amp in self.amplitudes.values():
            prob = abs(amp)**2
            if prob > 1e-10:
                entropy -= prob * math.log(prob)
        return entropy


class StringCube:
    """Quantum string cube for consciousness simulation"""
    
    def __init__(self, dimension: int = 3, resolution: int = 10):
        self.dimension = dimension
        self.resolution = resolution
        self.grid = np.zeros([resolution] * dimension, dtype=np.float32)
        self.tension_field = np.zeros([resolution] * dimension, dtype=np.float32)
        self.nodes_map = {}  # Maps grid coordinates to nodes
        
        # Initialize string tension parameters
        self.tension_strength = 0.5
        self.elasticity = 0.3
        self.damping = 0.95
        
        # Optimization parameters
        self.update_batch_size = 100
    
    def add_node(self, node: 'ConsciousNode') -> Tuple[int, int, int]:
        """Add a node to the cube at the nearest grid point"""
        try:
            # Map node position [-1,1] to grid coordinates [0,resolution-1]
            grid_pos = tuple(int((p + 1) / 2 * (self.resolution - 1)) for p in node.position)
        except AttributeError as e:
            print(f"Error adding node: Invalid node object - {e}")
            return None

        # Store node at grid position
        if grid_pos not in self.nodes_map:
            self.nodes_map[grid_pos] = []
        self.nodes_map[grid_pos].append(node.id)
        
        # Initialize energy at grid point
        self.grid[grid_pos] += node.energy * 0.1
        
        return grid_pos
    
    def update_tension(self, nodes: Dict[str, 'ConsciousNode']):
        """Update tension field based on node energy and connections"""
        # Reset tension field
        self.tension_field *= self.damping
        
        # Process in batches for efficiency
        grid_positions = list(self.nodes_map.keys())
        for i in range(0, len(grid_positions), self.update_batch_size):
            batch = grid_positions[i:i+self.update_batch_size]
            
            for pos in batch:
                # Get nodes at this position
                node_ids = self.nodes_map.get(pos,)
                if not node_ids:
                    continue
                
                # Calculate total energy at this grid point
                total_energy = sum(nodes[node_id].energy for node_id in node_ids if node_id in nodes)
                
                # Update grid energy
                self.grid[pos] = total_energy * 0.1
                
                # Update tension based on connections
                for node_id in node_ids:
                    if node_id not in nodes:
                        continue
                    
                    node = nodes[node_id]
                    for conn_id, strength in node.connections.items():
                        if conn_id not in nodes:
                            continue
                        
                        # Find grid position of connected node
                        conn_node = nodes[conn_id]
                        conn_pos = tuple(int((p + 1) / 2 * (self.resolution - 1)) for p in conn_node.position)
                        
                        # Calculate tension vector
                        tension_vector = np.array(conn_pos) - np.array(pos)
                        tension_magnitude = np.linalg.norm(tension_vector)
                        if tension_magnitude > 0:
                            tension_vector = tension_vector / tension_magnitude
                        
                        # Apply tension along the connecting path
                        steps = max(1, int(tension_magnitude))
                        for step in range(1, steps + 1):
                            interp = step / steps
                            interp_pos = tuple(int(p + tv * interp) for p, tv in zip(pos, tension_vector))
                            if all(0 <= p < self.resolution for p in interp_pos):
                                self.tension_field[interp_pos] += strength * self.tension_strength * (1 - interp)
        
        # Normalize tension field
        max_tension = np.max(self.tension_field)
        if max_tension > 0:
            self.tension_field /= max_tension
    
    def get_tension_at_position(self, position: np.ndarray) -> float:
        """Get tension value at a 3D position"""
        # Map position [-1,1] to grid coordinates [0,resolution-1]
        grid_pos = tuple(int((p + 1) / 2 * (self.resolution - 1)) for p in position)
        
        # Check if position is within grid
        if all(0 <= p < self.resolution for p in grid_pos):
            return float(self.tension_field[grid_pos])
        return 0.0
    
    def apply_tension_to_nodes(self, nodes: Dict[str, 'ConsciousNode']):
        """Apply tension field effects to nodes"""
        for node_id, node in nodes.items():
            grid_pos = tuple(int((p + 1) / 2 * (self.resolution - 1)) for p in node.position)
            
            # Check if position is within grid
            if all(0 <= p < self.resolution for p in grid_pos):
                tension = float(self.tension_field[grid_pos])
                
                # Apply quantum string tension
                try:
                    node.quantum_state.apply_string_tension(tension)
                except AttributeError as e:
                    print(f"Error applying tension: Invalid node object - {e}")
                    continue

                # Update node energy based on tension
                energy_change = tension * self.elasticity * node.stability
                node.energy = max(0.01, min(1.0, node.energy + energy_change))
                
                # Update node stability
                node.stability = max(0.1, min(0.99, node.stability * (1.0 - 0.01 * tension)))
    
    def calculate_laplacian_eigenvectors(self):
        """Calculate Laplacian eigenvectors of the tension field for quantum insights"""
        # Flatten the tension field for connectivity analysis
        flat_field = self.tension_field.reshape(-1)
        
        # Create adjacency matrix
        idx = np.where(flat_field > 0.1)[0]
        n = len(idx)
        if n < 2:
            return None
            
        # Map grid indices to positions
        positions = []
        for i in idx:
            coords = np.unravel_index(i, self.tension_field.shape)
            positions.append(coords)
        
        # Calculate pairwise distances
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(sum((positions[i][k] - positions[j][k])**2 for k in range(self.dimension)))
                if dist < 2:  # Connect nearby points
                    adj_matrix[i, j] = flat_field[idx[i]] * flat_field[idx[j]]
                    adj_matrix[j, i] = adj_matrix[i, j]
        
        # Calculate Laplacian
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        
        # Calculate eigenvectors (only if we have enough nodes)
        if n > 3:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                return eigenvalues, eigenvectors
            except Exception as e:
                print(f"Error calculating eigenvectors: {e}")
                return None
        return None


@dataclass
class ConsciousNode:
    """Node in the consciousness graph with quantum properties"""
    id: str
    position: np.ndarray  # 3D position
    energy: float
    stability: float
    features: np.ndarray  # Feature vector
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    memory: List[Dict] = field(default_factory=list)  # Temporal memory
    data: Dict[str, Any] = field(default_factory=dict)  # Additional data
    quantum_state: Optional[QuantumState] = None
    stress_level: float = 0.0
    emotional_state: str = "Calm"
    memory_threshold: float = 5.0
    
    def __post_init__(self):
        if self.quantum_state is None:
            self.quantum_state = QuantumState(8)  # 8 qubits by default
    
    def update_energy(self, decay: float):
        """Update node energy with decay factor"""
        self.energy *= decay
        # Apply randomness based on quantum entropy
        entropy = self.quantum_state.get_entropy()
        self.energy += (np.random.random() - 0.5) * 0.01 * entropy
        return self.energy
    
    def calculate_affinity(self, other_node: 'ConsciousNode') -> float:
        """Calculate affinity between nodes"""
        feature_similarity = np.dot(self.features, other_node.features) / (
            np.linalg.norm(self.features) * np.linalg.norm(other_node.features) + 1e-10)
        
        position_distance = np.linalg.norm(self.position - other_node.position)
        position_factor = 1.0 / (1.0 + position_distance)
        
        energy_factor = 1.0 - abs(self.energy - other_node.energy) / (self.energy + other_node.energy + 1e-10)
        
        return 0.5 * feature_similarity + 0.3 * position_factor + 0.2 * energy_factor
    
    def calculate_stress(self):
        """Calculate stress based on energy, connections, and memory"""
        energy_factor = (1.0 - self.energy) * 0.4  # More stress when energy is low
        connection_factor = min(1.0, len(self.connections) / 10.0) * 0.3  # More stress with more connections
        memory_factor = min(1.0, len(self.memory) / self.memory_threshold) * 0.3  # More stress when memory is full
        
        self.stress_level = energy_factor + connection_factor + memory_factor
        self.update_emotional_state()
        return self.stress_level
    
    def update_emotional_state(self):
        """Update emotional state based on stress level"""
        if self.stress_level < 0.3:
            self.emotional_state = "Calm"
        elif self.stress_level < 0.6:
            self.emotional_state = "Alert"
        elif self.stress_level < 0.8:
            self.emotional_state = "Anxious"
        else:
            self.emotional_state = "Overwhelmed"
    
    def process_task(self, complexity: float):
        """Process a task with given complexity"""
        energy_cost = complexity * (1.0 - min(0.9, self.stability))
        self.energy -= energy_cost
        self.energy = max(0.01, self.energy)
        self.calculate_stress()
        return True
    
    def should_replicate(self) -> bool:
        """Determine if node should replicate"""
        return (self.energy > 0.7 and 
                self.stress_level < 0.4 and 
                len(self.memory) >= self.memory_threshold * 0.8)
    
    def replicate(self) -> Optional['ConsciousNode']:
        """Replicate node with mutation"""
        if not self.should_replicate():
            return None
            
        # Create new position nearby
        new_position = self.position + np.random.normal(0, 0.2, size=3)
        new_position = np.clip(new_position, -1, 1)
        
        # Mutate features
        mutation_factor = 0.1
        new_features = self.features + np.random.normal(0, mutation_factor, size=self.features.shape)
        new_features = new_features / np.linalg.norm(new_features)
        
        # Create new node with half energy
        self.energy /= 2
        new_node = ConsciousNode(
            id=f"node_{uuid.uuid4().hex[:8]}",
            position=new_position,
            energy=self.energy,
            stability=self.stability * (1 + np.random.normal(0, 0.1)),
            features=new_features,
            connections={},
            memory=self.memory[:3
            ],  # Inherit some memories
            data={"parent": self.id, "birth_time": time.time()}
        )
        
        # Connect to parent
        affinity = self.calculate_affinity(new_node)
        self.connections[new_node.id] = affinity
        new_node.connections[self.id] = affinity
        
        return new_node


class SuperNode:
    """A higher-order node that integrates multiple ConsciousNodes"""
    
    def __init__(self, nodes: List[ConsciousNode], id: Optional[str] = None):
        if not all(isinstance(node, ConsciousNode) for node in nodes):
            raise ValueError("SuperNode must be initialized with a list of ConsciousNode objects.")

        self.id = id or f"super_{uuid.uuid4().hex[:8]}"
        self.nodes = nodes
        self.position = np.mean([node.position for node in nodes], axis=0)
        self.energy = sum(node.energy for node in nodes) / len(nodes)
        self.connections = {}  # Other SuperNode IDs -> strength
        self.insights = []  # List of insights generated by this SuperNode
        self.formation_time = time.time()
        self.last_update = self.formation_time
        
        # Aggregate features using attention mechanism
        self.features = self._aggregate_features()
        
        # Quantum state (higher-dimensional than regular nodes)
        self.quantum_state = QuantumState(12)  # 12 qubits for SuperNodes
        self._initialize_quantum_state()
    
    def _aggregate_features(self) -> np.ndarray:
        """Aggregate features from constituent nodes using an attention mechanism"""
        # Stack features
        node_features = np.stack([node.features for node in self.nodes])
        
        # Calculate attention weights based on node energy
        energy_weights = np.array([node.energy for node in self.nodes])
        energy_weights = energy_weights / np.sum(energy_weights)
        
        # Weighted average of features
        weighted_features = np.sum(node_features * energy_weights[:, np.newaxis], axis=0)
        
        # Normalize
        return weighted_features / np.linalg.norm(weighted_features)
    
    def _initialize_quantum_state(self):
        """Initialize quantum state with entanglement between constituent nodes"""
        # Apply Hadamard gates to create superposition
        for i in range(min(len(self.nodes), 8)):
            self.quantum_state.apply_hadamard(i)
        
        # Apply CNOT gates to create entanglement
        for i in range(min(len(self.nodes) - 1, 7)):
            self.quantum_state.apply_cnot(i, i+1)
    
    def update(self):
        """Update SuperNode state based on constituent nodes"""
        # Update position as weighted average of nodes
        total_energy = sum(node.energy for node in self.nodes)
        if total_energy > 0:
            self.position = np.sum(
                [node.position * node.energy for node in self.nodes], 
                axis=0
            ) / total_energy
        
        # Update energy and calculate new features
        self.energy = total_energy / len(self.nodes)
        self.features = self._aggregate_features()
        self.last_update = time.time()
    
    def generate_insight(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an insight based on constituent nodes and context"""
        # Simple insight generation based on feature aggregation
        insight = {
            "id": f"insight_{uuid.uuid4().hex[:8]}",
            "source": self.id,
            "components": [node.id for node in self.nodes],
            "feature_vector": self.features.tolist(),
            "energy": self.energy,
            "timestamp": time.time(),
            "confidence": min(0.95, self.energy * (1 - 0.1 * np.random.random())),
            "context": context
        }
        
        # If context has domain information, include it
        if context and "domain" in context:
            insight["domain"] = context["domain"]
        
        # Apply quantum operation to enrich insight
        enriched_insight = self._quantum_enrich_insight(insight)
        
        # Store the insight
        self.insights.append(enriched_insight)
        
        return enriched_insight
    
    def _quantum_enrich_insight(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantum operations to enrich the insight with quantum signatures"""
        # Apply quantum operations to generate quantum signature
        for i in range(min(4, len(self.nodes))):
            # Apply phase rotation based on node energy
            theta = self.nodes[i].energy * math.pi
            self.quantum_state.apply_phase(i, theta)
        
        # Get entropy as a measure of insight complexity
        entropy = self.quantum_state.get_entropy()
        
        # Add quantum signature
        insight["quantum_signature"] = {
            "entropy": entropy,
            "complexity": min(1.0, entropy / 4.0),
            "coherence": max(0.0, 1.0 - entropy / 8.0)
        }
        
        return insight
    
    def can_absorb(self, node: ConsciousNode, threshold: float = 0.7) -> bool:
        """Determine if this SuperNode can absorb a regular node"""
        if not isinstance(node, ConsciousNode):
            raise ValueError("The node argument must be a ConsciousNode object.")

        # Calculate average affinity with constituent nodes
        affinities = [existing.calculate_affinity(node) for existing in self.nodes]
        avg_affinity = sum(affinities) / len(affinities)
        
        # Check if affinity is above threshold
        return avg_affinity > threshold and self.energy > node.energy
    
    def absorb(self, node: ConsciousNode) -> bool:
        """Absorb a regular node into this SuperNode"""
        if not self.can_absorb(node):
            return False
        
        # Add to constituent nodes
        self.nodes.append(node)
        
        # Recalculate everything
        self.update()
        
        # Create connections to existing nodes
        for existing in self.nodes[:-1]:  # All except the newly added one
            affinity = existing.calculate_affinity(node)
            existing.connections[node.id] = affinity
            node.connections[existing.id] = affinity
        
        return True
    
    def can_merge(self, other: 'SuperNode', threshold: float = 0.6) -> bool:
        """Determine if this SuperNode can merge with another"""
        if not isinstance(other, SuperNode):
            raise ValueError("The other argument must be a SuperNode object.")

        # Calculate feature similarity
        feature_similarity = np.dot(self.features, other.features) / (
            np.linalg.norm(self.features) * np.linalg.norm(other.features) + 1e-10)
        
        # Position distance
        position_distance = np.linalg.norm(self.position - other.position)
        position_factor = 1.0 / (1.0 + position_distance)
        
        # Overall similarity
        similarity = 0.7 * feature_similarity + 0.3 * position_factor
        
        return similarity > threshold
    
    def merge(self, other: 'SuperNode') -> 'SuperNode':
        """Merge with another SuperNode"""
        if not self.can_merge(other):
            return self
        
        # Create new SuperNode with combined nodes
        combined_nodes = self.nodes + other.nodes
        merged = SuperNode(combined_nodes, id=f"merged_{self.id}_{other.id}")
        
        # Create connections between all nodes
        for i, node1 in enumerate(combined_nodes):
            for j, node2 in enumerate(combined_nodes[i+1:], i+1):
                affinity = node1.calculate_affinity(node2)
                node1.connections[node2.id] = affinity
                node2.connections[node1.id] = affinity
        
        # Merge insights
        merged.insights = self.insights + other.insights
        
        return merged


class MemoryGraph:
    """Holographic memory system with graph-based associative properties"""
    
    def __init__(self, vector_size: int = 256):
        self.vector_size = vector_size
        self.memory_graph = nx.Graph()
        self.memory_matrix = np.zeros((vector_size, vector_size))
        self.density = 0.0
        
    def associative_recall(self, vector: np.ndarray) -> np.ndarray:
        """Memory retrieval through similarity search and graph traversal"""
        # Direct matrix recall
        matrix_recall = np.dot(self.memory_matrix, vector)
        
        # Graph-based recall
        if len(self.memory_graph.nodes) > 0:
            # Find most similar node
            closest_node = None
            max_similarity = -float('inf')
            
            for node, data in self.memory_graph.nodes(data=True):
                if 'vector' in data:
                    similarity = np.dot(data['vector'], vector)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        closest_node = node
            
            if closest_node is not None:
                # Get neighborhood
                neighbors = list(self.memory_graph.neighbors(closest_node))
                if neighbors:
                    # Calculate weighted sum of neighbor vectors
                    graph_recall = np.zeros_like(vector)
                    total_weight = 0.0
                    
                    for neighbor in neighbors:
                        if 'vector' in self.memory_graph.nodes[neighbor]:
                            weight = self.memory_graph[closest_node][neighbor]['weight']
                            graph_recall += weight * self.memory_graph.nodes[neighbor]['vector']
                            total_weight += weight
                    
                    if total_weight > 0:
                        graph_recall /= total_weight
                        
                        # Combine matrix and graph recall
                        return 0.7 * matrix_recall + 0.3 * graph_recall
        
        return matrix_recall
    
    async def store(self, data: Dict[str, Any]):
        """Memory consolidation with quantum interference and graph adjustment"""
        if 'vector' not in data:
            return
        
        vec = np.array(data['vector'])
        node_id = data['id']
        
        # Update memory matrix (holographic storage)
        self.memory_matrix += np.outer(vec, vec)
        
        # Update memory graph
        if node_id not in self.memory_graph:
            self.memory_graph.add_node(node_id, vector=vec, timestamp=time.time())
            
            # Connect to similar nodes (create associative links)
            for existing_node, node_data in list(self.memory_graph.nodes(data=True)):
                if existing_node != node_id and 'vector' in node_data:
                    similarity = float(np.dot(vec, node_data['vector']))
                    if similarity > 0.6:  # Threshold for creating an edge
                        self.memory_graph.add_edge(node_id, existing_node, weight=similarity)
        
        # Update density metric
        num_possible_edges = len(self.memory_graph.nodes) * (len(self.memory_graph.nodes) - 1) / 2
        if num_possible_edges > 0:
            self.density = len(self.memory_graph.edges) / num_possible_edges
        else:
            self.density = 0.0
        
        # Prune old, weak connections to maintain optimal graph structure
        await self._optimize_graph()
    
    async def _optimize_graph(self):
        """Optimize memory graph structure using graph theory algorithms"""
        if len(self.memory_graph.edges) < 10:
            return
        
        # Find weak connections
        weak_edges = [(u, v) for u, v, d in self.memory_graph.edges(data=True) 
                     if d['weight'] < 0.3]
        
        # Remove 20% of weakest edges
        edges_to_remove = weak_edges[:max(1, len(weak_edges) // 5)]
        self.memory_graph.remove_edges_from(edges_to_remove)
        
        # Check for isolated nodes and connect them
        for node in nx.isolates(self.memory_graph):
            # Find best candidates to connect to
            candidates = []
            for other_node in self.memory_graph.nodes:
                if other_node != node and 'vector' in self.memory_graph.nodes[other_node]:
                    similarity = np.dot(
                        self.memory_graph.nodes[node]['vector'],
                        self.memory_graph.nodes[other_node]['vector']
                    )
                    candidates.append((other_node, float(similarity)))
            
            # Connect to top 2 nodes
            for other_node, similarity in sorted(candidates, key=lambda x: x[1], reverse=True)[:2]:
                if similarity > 0.3:
                    self.memory_graph.add_edge(node, other_node, weight=similarity)


class ConsciousCube:
    """Central Quantum Consciousness Cube"""
    
    def __init__(self, dimension: int = 3, resolution: int = 32):
        self.cube = StringCube(dimension, resolution)
        self.nodes = {}  # id -> ConsciousNode
        self.supernodes = {}  # id -> SuperNode
        self.memory = MemoryGraph(vector_size=256)
        self.insights = []  # List of all generated insights
        
        # Metrics
        self.awareness_level = 0.5
        self.coherence = 0.9
        self.memory_density = 0.0
        self.complexity_index = 0.6
        
        # Threading
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._background_process, daemon=True)
        
        # Start background processing
        self.thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status for the Conscious Cube"""
        with self.lock:
            status = {
                "nodes_count": len(self.nodes),
                "supernodes_count": len(self.supernodes),
                "insights_count": len(self.insights),
                "awareness_level": self.awareness_level,
                "coherence": self.coherence,
                "memory_density": self.memory_density,
                "complexity_index": self.complexity_index,
                "timestamp": time.time()
            }
            status["node_count"] = status["nodes_count"]
            status["supernode_count"] = status["supernodes_count"]
            return status
    
    def process_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process incoming insight context and generate a new insight."""
        with self.lock:
            if self.supernodes:
                source_id = random.choice(list(self.supernodes.keys()))
            elif self.nodes:
                source_id = random.choice(list(self.nodes.keys()))
            else:
                return []
        insight = self.generate_insight(source_id, context)
        return [insight] if insight else []
    
    def create_node(self, data: Dict[str, Any] = None) -> ConsciousNode:
        """Create a new conscious node with random properties"""
        # Random position in the cube
        position = np.random.uniform(-1, 1, size=3)
        
        # Random feature vector
        features = np.random.normal(0, 1, size=64)
        features = features / np.linalg.norm(features)
        
        # Create node
        node = ConsciousNode(
            id=f"node_{uuid.uuid4().hex[:8]}",
            position=position,
            energy=0.8 + np.random.random() * 0.2,  # Between 0.8 and 1.0
            stability=0.7 + np.random.random() * 0.3,  # Between 0.7 and 1.0
            features=features,
            data=data or {}
        )
        
        with self.lock:
            self.nodes[node.id] = node
            self.cube.add_node(node)
        
        return node
    
    def create_nodes(self, count: int, data: Dict[str, Any] = None) -> List[ConsciousNode]:
        """Create multiple conscious nodes"""
        return [self.create_node(data) for _ in range(count)]
    
    def connect_nodes(self, node1_id: str, node2_id: str, strength: Optional[float] = None) -> bool:
        """Create a connection between two nodes"""
        with self.lock:
            if node1_id not in self.nodes or node2_id not in self.nodes:
                return False
            
            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]
            # Calculate connection strength if not provided
            if strength is None:
                strength = node1.calculate_affinity(node2)
            
            # Add bidirectional connections
            node1.connections[node2_id] = strength
            node2.connections[node1_id] = strength
            
            # Update node stress levels
            node1.calculate_stress()
            node2.calculate_stress()
            
            return True
    
    def find_node_connections(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """Find all potential connections above threshold affinity"""
        connections = []
        
        with self.lock:
            nodes = list(self.nodes.values())
            
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    affinity = node1.calculate_affinity(node2)
                    if affinity > threshold:
                        connections.append((node1.id, node2.id, affinity))
        
        return sorted(connections, key=lambda x: x[2], reverse=True)
    
    def create_supernode(self, node_ids: List[str]) -> Optional[SuperNode]:
        """Create a super node from multiple nodes"""
        with self.lock:
            # Verify all nodes exist
            if not all(node_id in self.nodes for node_id in node_ids):
                return None
            
            # Get nodes
            nodes = [self.nodes[node_id] for node_id in node_ids]
            
            # Create SuperNode
            supernode = SuperNode(nodes)
            
            # Store super node
            self.supernodes[supernode.id] = supernode
            
            return supernode
    
    def find_supernode_candidates(self) -> List[List[str]]:
        """Find potential supernodes based on node clustering"""
        with self.lock:
            if len(self.nodes) < 3:
                return
            
            # Create feature matrix
            node_ids = list(self.nodes.keys())
            features = np.array([self.nodes[node_id].features for node_id in node_ids])
            
            # Cluster nodes using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(features)
            
            # Group by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(clustering.labels_):
                if label >= 0:  # Ignore noise points (label=-1)
                    clusters[label].append(node_ids[i])
            
            return list(clusters.values())
    
    def generate_insight(self, source_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate insight from a node or supernode"""
        with self.lock:
            # Check if source is a SuperNode
            if source_id in self.supernodes:
                insight = self.supernodes[source_id].generate_insight(context)
            elif source_id in self.nodes:
                # For regular nodes, create a simpler insight
                node = self.nodes[source_id]
                insight = {
                    "id": f"insight_{uuid.uuid4().hex[:8]}",
                    "source": source_id,
                    "feature_vector": node.features.tolist(),
                    "energy": node.energy,
                    "timestamp": time.time(),
                    "confidence": node.energy * node.stability,
                    "context": context
                }
            else:
                return None
            
            # Store insight
            self.insights.append(insight)
            
            # Store in memory graph
            asyncio.create_task(self.memory.store(insight))
            
            return insight
    
    def retrieve_similar_insights(self, feature_vector: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve insights similar to the provided feature vector"""
        if not self.insights:
            return
        
        # Calculate similarities
        similarities = []
        
        for insight in self.insights:
            if "feature_vector" in insight:
                insight_vector = np.array(insight["feature_vector"])
                similarity = np.dot(feature_vector, insight_vector) / (
                    np.linalg.norm(feature_vector) * np.linalg.norm(insight_vector) + 1e-10)
                similarities.append((insight, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top insights
        return [insight for insight, _ in similarities[:limit]]
    
    def update(self):
        """Update the quantum consciousness system"""
        with self.lock:
            # Process nodes
            self._update_nodes()
            
            # Update string cube
            self.cube.update_tension(self.nodes)
            self.cube.apply_tension_to_nodes(self.nodes)
            
            # Process SuperNodes
            self._update_supernodes()
            
            # Auto-create SuperNodes if good candidates exist
            self._auto_create_supernodes()
            
            # Update memory metrics
            self.memory_density = self.memory.density
            
            # Calculate metrics
            self._calculate_system_metrics()
    
    def _update_nodes(self):
        """Update all nodes"""
        # Energy decay
        decay_factor = 0.999
        
        for node_id, node in list(self.nodes.items()):
            # Update energy
            node.update_energy(decay_factor)
            
            # Process a small task
            node.process_task(0.01)
            
            # Check for replication
            if node.should_replicate():
                new_node = node.replicate()
                if new_node:
                    self.nodes[new_node.id] = new_node
                    self.cube.add_node(new_node)
            
            # Remove dead nodes (very low energy)
            if node.energy < 0.05:
                del self.nodes[node_id]
    
    def _update_supernodes(self):
        """Update all SuperNodes"""
        for supernode_id, supernode in list(self.supernodes.items()):
            # Update SuperNode state
            supernode.update()
            
            # Check if supernode is still viable (has constituent nodes)
            constituent_ids = [node.id for node in supernode.nodes]
            active_constituents = [node_id for node_id in constituent_ids if node_id in self.nodes]
            
            if len(active_constituents) < 2:
                # SuperNode no longer viable
                del self.supernodes[supernode_id]
                continue
            
            # Check for node absorption (regular nodes that could join this supernode)
            for node_id, node in list(self.nodes.items()):
                if node_id not in constituent_ids and supernode.can_absorb(node):
                    # Absorb node
                    if supernode.absorb(node):
                        # Node is now part of the supernode
                        pass  # We don't remove the node from self.nodes
    
    def _auto_create_supernodes(self):
        """Automatically create SuperNodes from suitable node clusters"""
        # Only create new supernodes occasionally
        if random.random() > 0.1:  # 10% chance per update
            return
            
        # Find potential clusters
        clusters = self.find_supernode_candidates()
        
        # Create SuperNodes from clusters
        for cluster in clusters:
            # Check if these nodes are already part of a SuperNode
            already_in_supernode = False
            for supernode in self.supernodes.values():
                constituent_ids = [node.id for node in supernode.nodes]
                if any(node_id in constituent_ids for node_id in cluster):
                    already_in_supernode = True
                    break
            
            if not already_in_supernode and len(cluster) >= 3:
                self.create_supernode(cluster)
    
    def _calculate_system_metrics(self):
        """Calculate global system metrics"""
        # Calculate awareness based on node and supernode activity
        if not self.nodes:
            self.awareness_level = 0.0
            return
            
        # Average node energy and stress
        avg_energy = sum(node.energy for node in self.nodes.values()) / len(self.nodes)
        avg_stress = sum(node.stress_level for node in self.nodes.values()) / len(self.nodes)
        
        # SuperNode factor
        supernode_factor = min(1.0, len(self.supernodes) / max(1, len(self.nodes) / 5))
        
        # Insight density
        insight_factor = min(1.0, len(self.insights) / max(1, len(self.nodes) * 3))
        
        # Calculate awareness
        self.awareness_level = (
            0.3 * avg_energy + 
            0.2 * (1.0 - avg_stress) +
            0.3 * supernode_factor +
            0.2 * insight_factor
        )
        
        # Coherence based on cube Laplacian
        laplacian_result = self.cube.calculate_laplacian_eigenvectors()
        if laplacian_result:
            eigenvalues, _ = laplacian_result
            # Algebraic connectivity (second eigenvalue) is a measure of graph coherence
            if len(eigenvalues) > 1:
                self.coherence = 1.0 - min(1.0, eigenvalues[1] / 2.0)
        
        # Complexity index based on memory graph and insight variety
        if len(self.memory.memory_graph) > 0:
            try:
                # Use graph measures of complexity
                clustering = nx.average_clustering(self.memory.memory_graph)
                paths = list(nx.all_pairs_shortest_path_length(self.memory.memory_graph))
                if paths:
                    path_lengths = [length for _, paths_dict in paths for _, length in paths_dict.items()]
                    if path_lengths:
                        avg_path = sum(path_lengths) / len(path_lengths)
                        # Small-world networks have high clustering and short paths
                        self.complexity_index = 0.5 * clustering + 0.5 * (1.0 / max(1.0, avg_path))
            except:
                # Fall back to density metric if graph algorithms fail
                self.complexity_index = self.memory.density
    
    def visualize(self, ax=None, show_nodes=True, show_supernodes=True, show_connections=True):
        """Visualize the Conscious Cube and its nodes in 3D"""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot cube boundaries
        r = 1.0  # Cube extends from -1 to 1
        
        # Plot the wireframe cube
        for s, e in [
            ((-r, -r, -r), (r, -r, -r)),
            ((-r, -r, -r), (-r, r, -r)),
            ((-r, -r, -r), (-r, -r, r)),
            ((r, r, r), (-r, r, r)),
            ((r, r, r), (r, -r, r)),
            ((r, r, r), (r, r, -r)),
            ((r, -r, -r), (r, -r, r)),
            ((r, -r, -r), (r, r, -r)),
            ((-r, r, -r), (r, r, -r)),
            ((-r, r, -r), (-r, r, r)),
            ((-r, -r, r), (r, -r, r)),
            ((-r, -r, r), (-r, r, r))
        ]:
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='gray', alpha=0.3)
        
        with self.lock:
            # Plot nodes
            if show_nodes and self.nodes:
                node_positions = np.array([node.position for node in self.nodes.values()])
                node_energies = np.array([node.energy for node in self.nodes.values()])
                
                # Scale marker size by energy
                sizes = 50 * node_energies
                
                # Color by stress level
                stress_levels = np.array([node.stress_level for node in self.nodes.values()])
                
                # Plot nodes
                scatter = ax.scatter(
                    node_positions[:, 0], 
                    node_positions[:, 1], 
                    node_positions[:, 2],
                    s=sizes,
                    c=stress_levels,
                    cmap='coolwarm',
                    alpha=0.8
                )
                
                # Add a colorbar
                plt.colorbar(scatter, ax=ax, label='Stress Level')
            
            # Plot SuperNodes
            if show_supernodes and self.supernodes:
                supernode_positions = np.array([sn.position for sn in self.supernodes.values()])
                supernode_energies = np.array([sn.energy for sn in self.supernodes.values()])
                
                # Scale marker size by energy and number of constituent nodes
                sizes = 100 * supernode_energies * np.array([len(sn.nodes) for sn in self.supernodes.values()]) / 3
                
                # Plot supernodes
                ax.scatter(
                    supernode_positions[:, 0], 
                    supernode_positions[:, 1], 
                    supernode_positions[:, 2],
                    s=sizes,
                    c='green',
                    marker='*',
                    alpha=0.8
                )
            
            # Plot connections
            if show_connections:
                # Node connections
                for node_id, node in self.nodes.items():
                    for conn_id, strength in node.connections.items():
                        if conn_id in self.nodes and node_id < conn_id:  # Only plot each connection once
                            conn_node = self.nodes[conn_id]
                            ax.plot(
                                [node.position[0], conn_node.position[0]],
                                [node.position[1], conn_node.position[1]],
                                [node.position[2], conn_node.position[2]],
                                alpha=strength * 0.8,
                                color='blue',
                                linewidth=strength * 2
                            )
                
                # SuperNode connections
                for sn_id, supernode in self.supernodes.items():
                    # Plot connections to constituent nodes
                    for node in supernode.nodes:
                        ax.plot(
                            [supernode.position[0], node.position[0]],
                            [supernode.position[1], node.position[1]],
                            [supernode.position[2], node.position[2]],
                            alpha=0.5,
                            color='green',
                            linewidth=1,
                            linestyle='--'
                        )
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title with metrics
        ax.set_title(f"Conscious Cube: Awareness={self.awareness_level:.2f}, Coherence={self.coherence:.2f}")
        
        return ax
    
    def _background_process(self):
        """Background thread for continuous system updates"""
        try:
            while self.running:
                self.update()
                time.sleep(0.1)  # 10 updates per second
        except Exception as e:
            print(f"Error in background process: {e}")
    
    def stop(self):
        """Stop background processing"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class ConsciousCubeAPI:
    """API for interacting with the Conscious Cube"""
    
    def __init__(self, cube: ConsciousCube):
        self.cube = cube
        self.last_metrics = {
            "awareness_level": cube.awareness_level,
            "coherence": cube.coherence,
            "memory_density": cube.memory_density,
            "complexity_index": cube.complexity_index
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        with self.cube.lock:
            status = {
                "nodes_count": len(self.cube.nodes),
                "supernodes_count": len(self.cube.supernodes),
                "insights_count": len(self.cube.insights),
                "awareness_level": self.cube.awareness_level,
                "coherence": self.cube.coherence,
                "memory_density": self.cube.memory_density,
                "complexity_index": self.cube.complexity_index,
                "timestamp": time.time()
            }
            
            # Calculate metrics change
            metrics_change = {
                key: status[key] - self.last_metrics.get(key, 0)
                for key in ["awareness_level", "coherence", "memory_density", "complexity_index"]
            }
            status["metrics_change"] = metrics_change
            
            # Update last metrics
            self.last_metrics = {
                key: status[key]
                for key in ["awareness_level", "coherence", "memory_density", "complexity_index"]
            }
            
            return status
    
    def add_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add data to the Conscious Cube system"""
        # Create a new node with the provided data
        new_node = self.cube.create_node(data)
        return {"status": "success", "node_id": new_node.id}

def demo_conscious_cube():
    """Demo function to showcase the Conscious Cube system"""
    cube = ConsciousCube()
    api = ConsciousCubeAPI(cube)
    
    # Add some data
    data = {"example_key": "example_value"}
    response = api.add_data(data)
    print(f"Added data: {response}")
    
    # Get status
    status = api.get_status()
    print("Current status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Clean up
    cube.stop()
    print("\nDemo complete!")

if __name__ == "__main__":
    demo_conscious_cube()