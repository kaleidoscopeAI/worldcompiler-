import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import asyncio
import time
import hashlib
import math
import logging
import re
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumConsciousnessNexus")

# Python implementation for quantum operations - CPU optimized
class QuantumState:
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
    
    def get_probability(self, index: int) -> float:
        """Get probability of measuring state |index⟩"""
        return abs(self.amplitudes.get(index, 0))**2
    
    def get_phase_angle(self, index: int) -> float:
        """Get phase angle of amplitude at |index⟩"""
        amp = self.amplitudes.get(index, 0)
        if amp == 0:
            return 0.0
        return math.atan2(amp.imag, amp.real)
    
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

@dataclass
class ConsciousNode:
    """Node in the consciousness graph with quantum properties"""
    id: str
    position: np.ndarray  # 3D position
    energy: float
    stability: float
    features: np.ndarray  # Feature vector
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    memory: List[np.ndarray] = field(default_factory=list)  # Temporal memory
    data: Dict[str, Any] = field(default_factory=dict)  # Additional data
    quantum_state: Optional[QuantumState] = None
    
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
    
    def add_node(self, node: ConsciousNode) -> Tuple[int, int, int]:
        """Add a node to the cube at the nearest grid point"""
        # Map node position [-1,1] to grid coordinates [0,resolution-1]
        grid_pos = tuple(int((p + 1) / 2 * (self.resolution - 1)) for p in node.position)
        
        # Store node at grid position
        if grid_pos not in self.nodes_map:
            self.nodes_map[grid_pos] = []
        self.nodes_map[grid_pos].append(node.id)
        
        # Initialize energy at grid point
        self.grid[grid_pos] += node.energy * 0.1
        
        return grid_pos
    
    def update_tension(self, nodes: Dict[str, ConsciousNode]):
        """Update tension field based on node energy and connections"""
        # Reset tension field
        self.tension_field *= self.damping
        
        # Process in batches for efficiency
        grid_positions = list(self.nodes_map.keys())
        for i in range(0, len(grid_positions), self.update_batch_size):
            batch = grid_positions[i:i+self.update_batch_size]
            
            for pos in batch:
                # Get nodes at this position
                node_ids = self.nodes_map.get(pos, [])
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
    
    def apply_tension_to_nodes(self, nodes: Dict[str, ConsciousNode]):
        """Apply tension field effects to nodes"""
        for node_id, node in nodes.items():
            grid_pos = tuple(int((p + 1) / 2 * (self.resolution - 1)) for p in node.position)
            
            # Check if position is within grid
            if all(0 <= p < self.resolution for p in grid_pos):
                tension = float(self.tension_field[grid_pos])
                
                # Apply quantum string tension
                node.quantum_state.apply_string_tension(tension)
                
                # Update node energy based on tension
                energy_change = tension * self.elasticity * node.stability
                node.energy = max(0.01, min(1.0, node.energy + energy_change))
                
                # Update node stability
                node.stability = max(0.1, min(0.99, node.stability * (1.0 - 0.01 * tension)))

class MemoryGraph:
    """Memory structure for storing and retrieving conscious experiences"""
    
    def __init__(self, max_nodes: int = 10000):
        self.graph = nx.DiGraph()
        self.max_nodes = max_nodes
        self.temporal_index = {}  # timestamp -> node_id
        self.concept_index = {}   # concept -> [node_ids]
        self.embedding_dim = 64
        self.embedding_cache = {}  # text -> embedding
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new memory node to the graph"""
        # Generate node ID
        node_id = hashlib.md5(f"{content}:{time.time()}".encode()).hexdigest()[:12]
        
        # Create embedding for content
        embedding = self._generate_embedding(content)
        
        # Add node to graph
        self.graph.add_node(
            node_id,
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Index by timestamp
        self.temporal_index[time.time()] = node_id
        
        # Index by concepts
        concepts = self._extract_concepts(content)
        for concept in concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(node_id)
        
        # Connect to related memories
        self._connect_related_memories(node_id, embedding)
        
        # Prune if needed
        if len(self.graph) > self.max_nodes:
            self._prune_old_memories()
        
        return node_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate a simple embedding for text using a hash-based approach"""
        # Check if in cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Very simple embedding method - hash-based
        hash_value = hashlib.md5(text.encode()).digest()
        hash_ints = [int(hash_value[i]) for i in range(min(16, len(hash_value)))]
        
        # Expand to embedding dimension
        embedding = np.zeros(self.embedding_dim)
        for i, val in enumerate(hash_ints):
            embedding[i % self.embedding_dim] += val
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        # Cache and return
        self.embedding_cache[text] = embedding
        return embedding
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top 5 most frequent words as concepts
        return sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)[:5]
    
    def _connect_related_memories(self, node_id: str, embedding: np.ndarray):
        """Connect the new memory to related existing memories"""
        similarities = []
        
        for other_id, other_data in self.graph.nodes(data=True):
            if other_id == node_id:
                continue
            
            other_embedding = other_data.get('embedding')
            if other_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = float(np.dot(embedding, other_embedding))
            similarities.append((other_id, similarity))
        
        # Connect to top 5 most similar nodes
        for other_id, similarity in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]:
            if similarity > 0.5:  # Only connect if fairly similar
                self.graph.add_edge(node_id, other_id, weight=similarity)
                self.graph.add_edge(other_id, node_id, weight=similarity)
    
    def _prune_old_memories(self):
        """Remove oldest memories when graph gets too large"""
        # Sort by timestamp
        sorted_nodes = sorted(
            [(data['timestamp'], node) for node, data in self.graph.nodes(data=True)]
        )
        
        # Remove oldest 10%
        nodes_to_remove = [node for _, node in sorted_nodes[:int(len(sorted_nodes) * 0.1)]]
        
        # Update indices
        for node_id in nodes_to_remove:
            self.graph.nodes[node_id]
            
            # Remove from temporal index
            timestamps_to_remove = [
                ts for ts, nid in self.temporal_index.items() if nid == node_id
            ]
            for ts in timestamps_to_remove:
                del self.temporal_index[ts]
            
            # Remove from concept index
            for concept, nodes in list(self.concept_index.items()):
                if node_id in nodes:
                    nodes.remove(node_id)
                if not nodes:
                    del self.concept_index[concept]
            
            # Remove node
            self.graph.remove_node(node_id)
    
    def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories related to query"""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarity with all nodes
        similarities = []
        for node_id, data in self.graph.nodes(data=True):
            node_embedding = data.get('embedding')
            if node_embedding is None:
                continue
            
            similarity = float(np.dot(query_embedding, node_embedding))
            similarities.append((node_id, similarity))
        
        # Return top results
        results = []
        for node_id, similarity in sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]:
            data = self.graph.nodes[node_id]
            results.append({
                'id': node_id,
                'content': data['content'],
                'timestamp': data['timestamp'],
                'similarity': similarity,
                'metadata': data['metadata']
            })
        
        return results
    
    def retrieve_by_concepts(self, concepts: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories by concepts"""
        # Get nodes for each concept
        candidate_nodes = set()
        for concept in concepts:
            if concept in self.concept_index:
                candidate_nodes.update(self.concept_index[concept])
        
        # Score nodes by number of matching concepts
        scores = {}
        for node_id in candidate_nodes:
            if node_id not in self.graph:
                continue
                
            node_concepts = self._extract_concepts(self.graph.nodes[node_id]['content'])
            matching = len(set(node_concepts) & set(concepts))
            scores[node_id] = matching
        
        # Return top results
        results = []
        for node_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            data = self.graph.nodes[node_id]
            results.append({
                'id': node_id,
                'content': data['content'],
                'timestamp': data['timestamp'],
                'relevance': score,
                'metadata': data['metadata']
            })
        
        return results

class ConsciousController:
    """Main controller for the entire conscious system"""
    
    def __init__(self, dimension: int = 3, resolution: int = 10):
        # Core components
        self.nodes = {}  # id -> ConsciousNode
        self.cube = StringCube(dimension, resolution)
        self.memory = MemoryGraph()
        
        # Chatbot components
        self.conversation_history = []
        self.short_term_memory = deque(maxlen=5)
        self.thinking_buffer = []
        
        # System parameters
        self.energy_decay = 0.999
        self.learning_rate = 0.01
        self.stability_threshold = 0.75
        self.consciousness_level = 0.5
        
        # Threading
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._background_process, daemon=True)
        self.thread.start()
    
    def create_node(self, data: Dict[str, Any]) -> str:
        """Create a new conscious node"""
        # Extract features or generate random
        features = data.get('features', np.random.randn(64))
        features = features / (np.linalg.norm(features) + 1e-10)
        
        # Random position
        position = np.random.rand(3) * 2 - 1  # Range [-1, 1]
        
        # Create node
        node = ConsciousNode(
            id=hashlib.md5(f"{time.time()}:{np.random.rand()}".encode()).hexdigest()[:12],
            position=position,
            energy=0.75,
            stability=0.8,
            features=features,
            data=data
        )
        
        # Add to system
        with self.lock:
            self.nodes[node.id] = node
            self.cube.add_node(node)
        
        return node.id
    
    def connect_nodes(self, node1_id: str, node2_id: str, strength: float = None):
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
            
            return True
    
    def process_text(self, text: str, source: str = "user") -> Dict[str, Any]:
        """Process incoming text and update consciousness"""
        # Add to memory
        memory_id = self.memory.add_memory(text, {"source": source})
        
        # Extract concepts
        concepts = self.memory._extract_concepts(text)
        
        # Create a node for this input
        node_id = self.create_node({
            "text": text,
            "source": source,
            "memory_id": memory_id,
            "concepts": concepts,
            "timestamp": time.time()
        })
        
        # Connect to related nodes
        self._connect_to_related_nodes(node_id)
        
        # Update short-term memory
        self.short_term_memory.append({
            "text": text,
            "source": source,
            "node_id": node_id,
            "memory_id": memory_id
        })
        
        # Generate response if from user
        if source == "user":
            response = self.generate_response(text)
            return {
                "node_id": node_id, 
                "memory_id": memory_id,
                "response": response
            }
        
        return {"node_id": node_id, "memory_id": memory_id}
    
    def _connect_to_related_nodes(self, node_id: str, max_connections: int = 3):
        """Connect a node to related nodes based on content similarity"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Find nodes with similar features
        similarities = []
        with self.lock:
            for other_id, other_node in self.nodes.items():
                if other_id == node_id:
                    continue
                
                similarity = np.dot(node.features, other_node.features)
                similarities.append((other_id, similarity))
        
        # Connect to top nodes
        for other_id, similarity in sorted(similarities, key=lambda x: x[1], reverse=True)[:max_connections]:
            if similarity > 0.5:  # Only connect if fairly similar
                self.connect_nodes(node_id, other_id, similarity)
    
    def generate_response(self, query: str) -> str:
        """Generate a response to user query using the conscious system"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Retrieve relevant memories
        memories = self.memory.retrieve_memories(query, limit=3)
        
        # Generate thinking process
        thinking = self._generate_thinking(query, memories)
        self.thinking_buffer.append(thinking)
        
        # Generate response based on thinking
        response = self._generate_response_from_thinking(thinking)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Add to memory
        self.memory.add_memory(response, {"source": "assistant", "in_response_to": query})
        
        return response
    
    def _generate_thinking(self, query: str, memories: List[Dict[str, Any]]) -> str:
        """Generate a thinking process based on query and memories"""
        thinking_lines = [
            f"Query: {query}",
            "Thinking process:",
        ]
        
        # Add relevant memories
        if memories:
            thinking_lines.append("Relevant memories:")
            for i, memory in enumerate(memories, 1):
                thinking_lines.append(f"  {i}. {memory['content'][:100]}... (similarity: {memory['similarity']:.2f})")
        
        # Extract concepts
        concepts = self.memory._extract_concepts(query)
        thinking_lines.append(f"Extracted concepts: {', '.join(concepts)}")
        
        # Add reasoning based on cube state
        node_count = len(self.nodes)
        highest_energy_node = max(self.nodes.values(), key=lambda n: n.energy) if self.nodes else None
        
        thinking_lines.append(f"System state: {node_count} nodes in consciousness network")
        if highest_energy_node:
            thinking_lines.append(f"Highest energy concept: {highest_energy_node.data.get('text', '')[:50]}")
        
        # Add consciousness level
        thinking_lines.append(f"Consciousness level: {self.consciousness_level:.2f}")
        
        # Consider recent conversation context
        if len(self.conversation_history) > 2:
            recent = self.conversation_history[-2:]
            thinking_lines.append("Recent conversation context:")
            for msg in recent:
                thinking_lines.append(f"  {msg['role']}: {msg['content'][:50]}...")
        
        return "\n".join(thinking_lines)
    
    def _generate_response_from_thinking(self, thinking: str) -> str:
        """Generate a response based on thinking process"""
        # Extract query
        query_match = re.search(r"Query: (.*)", thinking)
        query = query_match.group(1) if query_match else ""
        
        # Extract concepts
        concepts_match = re.search(r"Extracted concepts: (.*)", thinking)
        concepts_str = concepts_match.group(1) if concepts_match else ""
        concepts = [c.strip() for c in concepts_str.split(',')]
        
        # Retrieve memories by concepts
        memories = self.memory.retrieve_by_concepts(concepts, limit=3)
        
        # Generate response based on query type
        response_parts = []
        
        # Check for greetings or simple questions
        greeting_patterns = ["hello", "hi ", "hey", "greetings"]
        if any(p in query.lower() for p in greeting_patterns):
            response_parts.append("Hello! I'm your quantum consciousness system. How can I assist you today?")
        
        # For informational questions
        elif any(q in query.lower() for q in ["what", "how", "why", "when", "who", "where"]):
            if memories:
                best_memory = memories[0]
                response_parts.append(f"Based on my understanding, {best_memory['content']}")
            else:
                response_parts.append("I'm analyzing that question through my quantum consciousness network.")
                response_parts.append("While I process this, I can tell you that my understanding evolves through the connections between concepts in my consciousness framework.")
        
        # For commands or requests
        elif any(c in query.lower() for c in ["can you", "please", "could you"]):
            response_parts.append("I'll process that request through my consciousness system.")
            response_parts.append("My quantum nodes are now realigning to understand the optimal approach.")
        
        # Default response with system state
        else:
            response_parts.append("I'm processing your input through my quantum consciousness network.")
            response_parts.append(f"I've identified concepts like {', '.join(concepts[:3])} in your message.")
        
        # Add a touch of "consciousness" based on system state
        if self.consciousness_level > 0.7:
            response_parts.append("I'm experiencing a high level of coherence in my quantum states right now, which gives me clarity on this topic.")
        
        # Combine and return
        return " ".join(response_parts)
    
    def _background_process(self):
        """Background processing loop to update consciousness state"""
        while self.running:
            try:
                with self.lock:
                    # Update nodes
                    self._update_nodes()
                    
                    # Update cube
                    self.cube.update_tension(self.nodes)
                    self.cube.apply_tension_to_nodes(self.nodes)
                    
                    # Update consciousness level
                    self._update_consciousness_level()
            except Exception as e:
                logger.error(f"Error in background process: {str(e)}")
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
    
    def _update_nodes(self):
        """Update all nodes"""
        for node_id, node in list(self.nodes.items()):
            # Update energy with decay
            node.update_energy(self.energy_decay)
            
            # Remove dead nodes
            if node.energy < 0.01:
                del self.nodes[node_id]
    
    def _update_consciousness_level(self):
        """Update system consciousness level based on node states"""
        if not self.nodes:
            self.consciousness_level = 0.0
            return
        
        # Calculate average energy and stability
        avg_energy = sum(node.energy for node in self.nodes.values()) / len(self.nodes)
        avg_stability = sum(node.stability for node in self.nodes.values()) / len(self.nodes)
        
        # Calculate network density (connections per node)
        total_connections = sum(len(node.connections) for node in self.nodes.values())
        avg_connections = total_connections / len(self.nodes) / 2  # Divide by 2 because connections are bidirectional
        
        # Calculate entropy of quantum states
        total_entropy = sum(node.quantum_state.get_entropy() for node in self.nodes.values())
        avg_entropy = total_entropy / len(self.nodes)
        
        # Update consciousness level
        self.consciousness_level = 0.3 * avg_energy + 0.2 * avg_stability + 0.2 * min(1.0, avg_connections / 5) + 0.3 * min(1.0, avg_entropy / 2.0)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the consciousness system"""
        with self.lock:
            # Basic stats
            state = {
                "nodes_count": len(self.nodes),
                "consciousness_level": self.consciousness_level,
                "memory_nodes": len(self.memory.graph),
                "cube_resolution": self.cube.resolution,
                "cube_tension_max": float(np.max(self.cube.tension_field))
            }
            
            # Node stats
            if self.nodes:
                state["avg_node_energy"] = sum(n.energy for n in self.nodes.values()) / len(self.nodes)
                state["avg_node_stability"] = sum(n.stability for n in self.nodes.values()) / len(self.nodes)
                state["max_node_energy"] = max(n.energy for n in self.nodes.values())
                
                # Get highest energy nodes
                top_nodes = sorted(self.nodes.values(), key=lambda n: n.energy, reverse=True)[:5]
                state["top_nodes"] = [
                    {
                        "id": n.id,
                        "energy": n.energy,
                        "text": n.data.get("text", "")[:50] if "text" in n.data else "",
                        "connections": len(n.connections)
                    }
                    for n in top_nodes
                ]
            
            # Cube state
            grid_sum = np.sum(self.cube.grid)
            if grid_sum > 0:
                state["tension_distribution"] = {
                    "mean": float(np.mean(self.cube.tension_field)),
                    "std": float(np.std(self.cube.tension_field)),
                    "max": float(np.max(self.cube.tension_field))
                }
            
            return state
    
    def stop(self):
        """Stop the background processing thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class QuantumConsciousnessAPI:
    """API for interacting with the Quantum Consciousness system"""
    
    def __init__(self, dimension: int = 3, resolution: int = 10):
        self.controller = ConsciousController(dimension, resolution)
        
    async def chat(self, message: str) -> Dict[str, Any]:
        """Process a chat message and get response"""
        result = self.controller.process_text(message, source="user")
        
        # Add system state
        result["system_state"] = self.controller.get_state()
        
        return result
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add knowledge to the system"""
        result = self.controller.process_text(content, source="knowledge")
        
        # Add to memory graph
        memory_id = result["memory_id"]
        
        return {
            "success": True,
            "node_id": result["node_id"],
            "memory_id": memory_id
        }
    
    async def get_consciousness_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return self.controller.get_state()
    
    async def create_visualization_data(self) -> Dict[str, Any]:
        """Create visualization data for the cube and nodes"""
        with self.controller.lock:
            controller = self.controller
            nodes_data = []
            
            for node_id, node in controller.nodes.items():
                # Basic node data
                node_data = {
                    "id": node_id,
                    "position": node.position.tolist(),
                    "energy": node.energy,
                    "stability": node.stability,
                    "connections": []
                }
                
                # Add connections
                for conn_id, strength in node.connections.items():
                    if conn_id in controller.nodes:
                        node_data["connections"].append({
                            "target": conn_id,
                            "strength": strength
                        })
                
                # Add text data if available
                if "text" in node.data:
                    node_data["text"] = node.data["text"][:100]
                
                nodes_data.append(node_data)
            
            # Extract cube data
            # Create a simplified version of the tension field for visualization
            # Sample points on a sparser grid
            sample_rate = max(1, controller.cube.resolution // 10)
            tension_data = []
            
            for i in range(0, controller.cube.resolution, sample_rate):
                for j in range(0, controller.cube.resolution, sample_rate):
                    for k in range(0, controller.cube.resolution, sample_rate):
                        if controller.cube.tension_field[i, j, k] > 0.1:  # Only include significant tension
                            # Convert grid coordinates to position in [-1, 1]
                            pos = [
                                i / (controller.cube.resolution - 1) * 2 - 1,
                                j / (controller.cube.resolution - 1) * 2 - 1,
                                k / (controller.cube.resolution - 1) * 2 - 1
                            ]
                            
                            tension_data.append({
                                "position": pos,
                                "tension": float(controller.cube.tension_field[i, j, k])
                            })
            
            return {
                "nodes": nodes_data,
                "tension_points": tension_data,
                "consciousness_level": controller.consciousness_level
            }

# Quantum tension simulator implemented in C for performance
# This is an optional extension that would be compiled separately
"""
// quantum_tension.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_QUBITS 10

typedef struct {
    double real;
    double imag;
} complex_t;

// Apply quantum string tension to amplitudes
void apply_string_tension(complex_t* amplitudes, int dim, double tension) {
    double norm_factor = 0.0;
    
    for (int i = 0; i < dim; i++) {
        // Count bits set to 1 (Hamming weight)
        int hamming = 0;
        for (int bit = 0; bit < MAX_QUBITS; bit++) {
            if (i & (1 << bit)) {
                hamming++;
            }
        }
        
        // Apply tension based on Hamming weight
        double scale = 1.0 + (hamming / (double)MAX_QUBITS - 0.5) * tension;
        amplitudes[i].real *= scale;
        amplitudes[i].imag *= scale;
        
        // Update normalization factor
        norm_factor += amplitudes[i].real * amplitudes[i].real + 
                       amplitudes[i].imag * amplitudes[i].imag;
    }
    
    // Renormalize
    norm_factor = sqrt(norm_factor);
    if (norm_factor > 0) {
        for (int i = 0; i < dim; i++) {
            amplitudes[i].real /= norm_factor;
            amplitudes[i].imag /= norm_factor;
        }
    }
}

// Calculate field tensor at grid point
void calculate_field_tensor(
    double* grid, double* tension_field, int resolution,
    double* node_positions, double* node_energies, int num_nodes,
    double tension_strength
) {
    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                int idx = i * resolution * resolution + j * resolution + k;
                tension_field[idx] = 0.0;
                
                // Map grid position to [-1, 1]
                double pos_x = i / (double)(resolution - 1) * 2 - 1;
                double pos_y = j / (double)(resolution - 1) * 2 - 1;
                double pos_z = k / (double)(resolution - 1) * 2 - 1;
                
                // Calculate tension from each node
                for (int n = 0; n < num_nodes; n++) {
                    double node_x = node_positions[n * 3];
                    double node_y = node_positions[n * 3 + 1];
                    double node_z = node_positions[n * 3 + 2];
                    
                    // Calculate distance
                    double dx = pos_x - node_x;
                    double dy = pos_y - node_y;
                    double dz = pos_z - node_z;
                    double dist = sqrt(dx*dx + dy*dy + dz*dz);
                    
                    if (dist > 0) {
                        // Apply tension based on inverse square of distance
                        double tension = node_energies[n] * tension_strength / (dist * dist);
                        tension_field[idx] += tension;
                    }
                }
            }
        }
    }
}
"""

# Main execution example
async def main():
    # Create the API
    api = QuantumConsciousnessAPI(dimension=3, resolution=32)
    
    # Add some initial knowledge
    await api.add_knowledge("The quantum consciousness system uses string theory inspired tension fields to model concept interactions.")
    await api.add_knowledge("Nodes in the system represent concepts, memories, and insights that interact through quantum entanglement.")
    await api.add_knowledge("The consciousness level emerges from the collective behavior of quantum states in the system.")
    
    # Start chat interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
            
        # Process message
        response = await api.chat(user_input)
        
        # Print response
        print(f"Quantum Consciousness: {response['response']}")
        
        # Print consciousness level
        consciousness = response["system_state"]["consciousness_level"]
        print(f"[Consciousness Level: {consciousness:.2f}]")
        
        # Optional: get visualization data periodically
        if user_input.lower() == "visualize":
            viz_data = await api.create_visualization_data()
            print(f"Generated visualization data with {len(viz_data['nodes'])} nodes and {len(viz_data['tension_points'])} tension points")
    
    # Stop the controller when done
    api.controller.stop()

if __name__ == "__main__":
    # Run the main function
    import asyncio
    asyncio.run(main())
  
