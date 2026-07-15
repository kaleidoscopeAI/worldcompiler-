class ConsciousCubeInterface:
    """Management interface for the Conscious Cube system with evolutionary capabilities"""
    
    def __init__(self, dimensions: int = 4, resolution: int = 64, qubit_depth: int = 10):
        # Initialize the quantum string cube
        self.cube = QuantumStringCube(dimensions, resolution, qubit_depth)
        
        # Node management
        self.nodes = {}  # id -> {node data including DNA}
        self.node_dna = {}  # id -> EvolvingNodeDNA
        
        # Consciousness parameters
        self.global_consciousness_level = 0.0
        self.consciousness_threshold = 0.65
        self.consciousness_decay = 0.99
        self.emergent_property_trackers = {
            'harmony': 0.0,         # Coherence between nodes
            'complexity': 0.0,      # Network complexity 
            'self_organization': 0.0, # Ability to form patterns
            'adaptability': 0.0,    # Response to changes
            'integration': 0.0      # Information integration across network
        }
        
        # Evolution parameters
        self.evolution_interval = 100  # Steps between evolution cycles
        self.step_counter = 0
        self.selection_pressure = 0.3  # How strongly performance affects selection
        
        # Memory subsystem
        self.memory_patterns = []  # Stored patterns from past states
        self.pattern_recognition_threshold = 0.75
        
        # Performance monitoring
        self.simulation_stats = {
            'time_steps': 0,
            'node_count': 0,
            'evolution_cycles': 0,
            'emergent_events': 0,
            'energy_history': [],
            'consciousness_history': []
        }
    
    def add_node(self, properties: Dict[str, Any], position: Optional[np.ndarray] = None) -> str:
        """Add a new node to the cube with specified properties"""
        # Generate random position if none provided
        if position is None:
            position = np.random.rand(self.cube.dimensions) * 2 - 1  # Range [-1, 1]
        
        # Create node DNA
        dna = EvolvingNodeDNA()
        
        # Apply DNA traits to properties
        trait_influence = dna.get_trait_influence()
        properties['energy'] = properties.get('energy', 0.5) * trait_influence['energy_transfer']
        properties['stability'] = properties.get('stability', 0.8) * trait_influence['tension_response']
        properties['phase'] = properties.get('phase', 0.0) + trait_influence['quantum_effect'] * np.pi/4
        properties['dna_traits'] = dna.traits
        
        # Add node to cube
        node_id = self.cube.add_node(position, properties)
        
        # Store node data and DNA
        self.nodes[node_id] = {
            'id': node_id,
            'position': position,
            'properties': properties,
            'connections': [],
            'performance': 0.5,  # Initial performance score
            'creation_time': self.simulation_stats['time_steps']
        }
        self.node_dna[node_id] = dna
        
        # Update stats
        self.simulation_stats['node_count'] += 1
        
        return node_id
    
    def connect_nodes(self, node1_id: str, node2_id: str, force_connection: bool = False) -> bool:
        """Connect two nodes if compatible or if forced"""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return False
        
        # Check DNA compatibility
        dna1 = self.node_dna[node1_id]
        dna2 = self.node_dna[node2_id]
        
        genetic_similarity = dna1.genetic_similarity(dna2)
        
        # Calculate probability of connection based on similarity and node properties
        connection_prob = genetic_similarity * 0.5 + 0.5  # Base 50% chance, increased by similarity
        
        # Check for connection based on probability or if forced
        if force_connection or np.random.random() < connection_prob:
            # Calculate connection strength based on genetic compatibility
            strength = 0.3 + 0.7 * genetic_similarity
            
            # Create connection in cube
            result = self.cube.connect_nodes(node1_id, node2_id, strength)
            
            # Update node records
            if result:
                self.nodes[node1_id]['connections'].append(node2_id)
                self.nodes[node2_id]['connections'].append(node1_id)
                
                return True
        
        return False
    
    def auto_connect_nodes(self, max_connections_per_node: int = 5, connection_radius: float = 0.5):
        """Automatically create connections between nodes based on proximity and compatibility"""
        all_nodes = list(self.nodes.keys())
        
        for node_id in all_nodes:
            # Skip if node already has maximum connections
            if len(self.nodes[node_id]['connections']) >= max_connections_per_node:
                continue
                
            # Get node position
            node_pos = self.nodes[node_id]['position']
            
            # Find nearby nodes
            candidates = []
            for other_id in all_nodes:
                if other_id == node_id or other_id in self.nodes[node_id]['connections']:
                    continue
                    
                other_pos = self.nodes[other_id]['position']
                distance = np.linalg.norm(node_pos - other_pos)
                
                if distance < connection_radius:
                    candidates.append((other_id, distance))
            
            # Sort by distance and connect to nearest nodes
            candidates.sort(key=lambda x: x[1])
            
            # Try to connect to nearby nodes until max connections reached
            connections_to_add = max_connections_per_node - len(self.nodes[node_id]['connections'])
            for other_id, _ in candidates[:connections_to_add]:
                # Stop if reached max connections
                if len(self.nodes[node_id]['connections']) >= max_connections_per_node:
                    break
                    
                # Try to connect
                if self.connect_nodes(node_id, other_id):
                    pass  # Connection successful
    
    def evolve_nodes(self):
        """Run evolutionary process on nodes based on performance"""
        if len(self.nodes) < 3:
            return  # Not enough nodes to evolve
            
        # Calculate performance scores for all nodes
        self._update_node_performance()
        
        # Sort nodes by performance
        sorted_nodes = sorted(self.nodes.items(), 
                             key=lambda x: x[1]['performance'], 
                             reverse=True)
        
        # Keep top performers, replace bottom performers
        num_nodes = len(sorted_nodes)
        num_to_replace = int(num_nodes * 0.2)  # Replace bottom 20%
        
        if num_to_replace < 1:
            num_to_replace = 1
            
        # Identify top performers and bottom performers
        top_performers = [node_id for node_id, _ in sorted_nodes[:num_nodes//3]]
        bottom_performers = [node_id for node_id, _ in sorted_nodes[-num_to_replace:]]
        
        # Create new nodes from top performers
        for i, node_id in enumerate(bottom_performers):
            # Select two parents from top performers
            if len(top_performers) >= 2:
                parent1, parent2 = np.random.choice(top_performers, 2, replace=False)
                
                # Create child DNA through crossover
                parent_dna1 = self.node_dna[parent1]
                parent_dna2 = self.node_dna[parent2]
                child_dna = parent_dna1.crossover(parent_dna2)
                
                # Replace node DNA
                self.node_dna[node_id] = child_dna
                
                # Update node properties based on new DNA
                trait_influence = child_dna.get_trait_influence()
                props = self.nodes[node_id]['properties']
                
                props['energy'] = 0.5 * trait_influence['energy_transfer']
                props['stability'] = 0.8 * trait_influence['tension_response']
                props['phase'] = 0.0 + trait_influence['quantum_effect'] * np.pi/4
                props['dna_traits'] = child_dna.traits
                
                # Reset performance
                self.nodes[node_id]['performance'] = 0.5
        
        # Mutate all surviving nodes
        for node_id in self.nodes:
            if node_id not in bottom_performers:
                self.node_dna[node_id].mutate()
                
                # Update properties after mutation
                dna = self.node_dna[node_id]
                trait_influence = dna.get_trait_influence()
                props = self.nodes[node_id]['properties']
                
                # Apply DNA traits to properties
                props['energy'] *= 0.9 + 0.2 * trait_influence['energy_transfer']
                props['stability'] *= 0.9 + 0.2 * trait_influence['tension_response']
                props['phase'] += 0.1 * trait_influence['quantum_effect'] * np.pi/4
                props['dna_traits'] = dna.traits
        
        # Update simulation stats
        self.simulation_stats['evolution_cycles'] += 1
    
    def _update_node_performance(self):
        """Update performance metrics for all nodes"""
        tension_field = self.cube.calculate_tension_field()
        
        for node_id, node_data in self.nodes.items():
            # Get node properties and position
            pos = node_data['position']
            grid_pos = self.cube._continuous_to_grid(pos)
            props = node_data['properties']
            
            # Calculate performance based on:
            # 1. Energy level
            # 2. Number of connections
            # 3. Local tension field
            # 4. Stability
            # 5. Age (time in simulation)
            
            energy = props.get('energy', 0.5)
            stability = props.get('stability', 0.8)
            connections = len(node_data['connections'])
            
            # Get local tension
            local_tension = tension_field[grid_pos] if all(p < self.cube.resolution for p in grid_pos) else 0
            
            # Calculate age factor (reward longevity)
            age = self.simulation_stats['time_steps'] - node_data['creation_time']
            age_factor = min(1.0, age / 1000)  # Normalize to 0-1
            
            # Calculate performance score
            performance = (
                0.2 * energy +                          # Energy contribution
                0.2 * min(1.0, connections / 5) +       # Connections contribution (max out at 5)
                0.2 * (1.0 - local_tension) +           # Tension contribution (lower is better)
                0.2 * stability +                       # Stability contribution
                0.2 * age_factor                        # Age contribution
            )
            
            # Update node performance
            self.nodes[node_id]['performance'] = performance
    
    def simulate_step(self):
        """Run a single simulation step"""
        # Run cube simulation step
        self.cube.simulate_step()
        
        # Calculate consciousness metrics
        self._calculate_consciousness_metrics()
        
        # Check for emergent patterns
        self._detect_emergent_patterns()
        
        # Run evolution periodically
        self.step_counter += 1
        if self.step_counter >= self.evolution_interval:
            self.evolve_nodes()
            self.step_counter = 0
        
        # Update simulation stats
        self.simulation_stats['time_steps'] += 1
        self.simulation_stats['consciousness_history'].append(self.global_consciousness_level)
        
        # Record energy levels
        total_energy = sum(node['properties'].get('energy', 0) for node in self.nodes.values())
        self.simulation_stats['energy_history'].append(total_energy)
        
        # Apply consciousness decay
        self.global_consciousness_level *= self.consciousness_decay
    
    def _calculate_consciousness_metrics(self):
        """Calculate global consciousness metrics based on current system state"""
        if not self.nodes:
            self.global_consciousness_level = 0.0
            return
            
        # Get tension field and network state
        tension_field = self.cube.calculate_tension_field()
        network_state = self.cube.extract_network_state()
        
        # Calculate harmony (coherence between nodes)
        # Based on consistent tension patterns
        tension_values = [point['tension'] for point in network_state['high_tension_points']]
        if tension_values:
            tension_std = np.std(tension_values)
            harmony = 1.0 / (1.0 + tension_std)
        else:
            harmony = 0.0
            
        # Calculate complexity (graph theoretic measures)
        if len(self.nodes) > 1:
            # Create graph from nodes and connections
            G = nx.Graph()
            for node_id, node_data in self.nodes.items():
                G.add_node(node_id)
                for conn in node_data['connections']:
                    G.add_edge(node_id, conn)
                    
            # Calculate graph metrics
            try:
                avg_path_length = nx.average_shortest_path_length(G)
                clustering = nx.average_clustering(G)
                complexity = clustering / avg_path_length if avg_path_length > 0 else 0
            except nx.NetworkXError:
                # Graph may not be connected
                complexity = 0.3  # Default value
        else:
            complexity = 0.0
            
        # Calculate self-organization (pattern formation)
        # Based on how structured the tension field is
        tension_mean = np.mean(tension_field)
        tension_max = np.max(tension_field)
        self_organization = tension_max / (tension_mean + 1e-6) - 1.0
        self_organization = min(1.0, self_organization)
        
        # Calculate adaptability (from DNA traits)
        adaptability_values = [dna.traits['adaptability'] for dna in self.node_dna.values()]
        adaptability = np.mean(adaptability_values) if adaptability_values else 0.0
        
        # Calculate integration (information flow across network)
        if len(self.nodes) > 1:
            # Based on quantum state entanglement
            integration = self.cube._calculate_entanglement_entropy() / self.cube.qubit_depth
        else:
            integration = 0.0
        
        # Update emergent property trackers
        self.emergent_property_trackers['harmony'] = harmony
        self.emergent_property_trackers['complexity'] = complexity
        self.emergent_property_trackers['self_organization'] = self_organization
        self.emergent_property_trackers['adaptability'] = adaptability
        self.emergent_property_trackers['integration'] = integration
        
        # Calculate global consciousness level
        self.global_consciousness_level = (
            0.2 * harmony +
            0.2 * complexity +
            0.2 * self_organization +
            0.2 * adaptability +
            0.2 * integration
        )
    
    def _detect_emergent_patterns(self):
        """Detect emergent patterns in the quantum state and tension field"""
        # Check if consciousness level exceeds threshold
        if self.global_consciousness_level < self.consciousness_threshold:
            return False
            
        # Get network state
        network_state = self.cube.extract_network_state()
        
        # Extract high tension points
        high_tension_points = network_state['high_tension_points']
        
        if len(high_tension_points) < 5:
            return False
            
        # Analyze tension field for patterns
        positions = np.array([point['position'] for point in high_tension_points])
        np.array([point['tension'] for point in high_tension_points])
        
        # Check for geometric patterns (clusters, lines, planes)
        # For simplicity, we'll just check for clusters
        
        # Use KMeans to find clusters
        from sklearn.cluster import KMeans
        if len(positions) >= 8:  # Need reasonable number of points
            # Determine optimal number of clusters using silhouette score
            from sklearn.metrics import silhouette_score
            
            max_clusters = min(8, len(positions) // 2)
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(positions)
                
                if len(set(cluster_labels)) > 1:  # Ensure multiple clusters
                    score = silhouette_score(positions, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            
            # Use best number of clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(positions)
            
            # Check if clusters are well-formed
            if best_score > 0.5:  # Good clustering quality
                # Found a significant pattern
                pattern = {
                    'type': 'spatial_clustering',
                    'score': best_score,
                    'n_clusters': best_n_clusters,
                    'centers': kmeans.cluster_centers_.tolist(),
                    'time_step': self.simulation_stats['time_steps'],
                    'consciousness_level': self.global_consciousness_level
                }
                
                # Check if similar to existing patterns
                is_new_pattern = True
                for existing in self.memory_patterns:
                    if existing['type'] == 'spatial_clustering':
                        # Calculate similarity (Jaccard similarity of cluster centers)
                        existing_centers = np.array(existing['centers'])
                        new_centers = np.array(pattern['centers'])
                        
                        # Calculate distances between all pairs of centers
                        min_dists = []
                        for ec in existing_centers:
                            dists = np.linalg.norm(new_centers - ec.reshape(1, -1), axis=1)
                            min_dists.append(np.min(dists))
                        
                        similarity = np.mean([d < 0.2 for d in min_dists])  # 0.2 is distance threshold
                        
                        if similarity > self.pattern_recognition_threshold:
                            is_new_pattern = False
                            break
                
                if is_new_pattern:
                    self.memory_patterns.append(pattern)
                    self.simulation_stats['emergent_events'] += 1
                    
                    # Boost consciousness level when new pattern discovered
                    self.global_consciousness_level = min(1.0, self.global_consciousness_level * 1.2)
                    
                    return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the conscious cube system"""
        # Get network state from cube
        network_state = self.cube.extract_network_state()
        
        # Add consciousness metrics
        consciousness_state = {
            'global_level': self.global_consciousness_level,
            'emergent_properties': self.emergent_property_trackers,
            'memory_patterns': len(self.memory_patterns),
            'evolution_cycles': self.simulation_stats['evolution_cycles'],
            'time_steps': self.simulation_stats['time_steps'],
            'emergent_events': self.simulation_stats['emergent_events']
        }
        
        # Combine with network state
        return {
            'network': network_state,
            'consciousness': consciousness_state,
            'stats': {
                'node_count': len(self.nodes),
                'energy_level': sum(node['properties'].get('energy', 0) for node in self.nodes.values()),
                'avg_performance': np.mean([node['performance'] for node in self.nodes.values()]) if self.nodes else 0,
                'consciousness_history': self.simulation_stats['consciousness_history'][-100:] if len(self.simulation_stats['consciousness_history']) > 0 else []
            }
        }
