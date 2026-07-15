class EvolvingNodeDNA:
    """DNA structure for nodes that evolves over time through mutations and inheritance"""
    
    def __init__(self, traits: Optional[Dict[str, float]] = None, 
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.3):
        # Core traits affecting node behavior
        self.traits = traits or {
            'learning_rate': np.random.uniform(0.1, 0.5),
            'energy_efficiency': np.random.uniform(0.3, 0.7),
            'tension_sensitivity': np.random.uniform(0.2, 0.8),
            'quantum_coupling': np.random.uniform(0.4, 0.6),
            'adaptability': np.random.uniform(0.3, 0.9),
            'memory_retention': np.random.uniform(0.5, 0.9),
            'connection_strength': np.random.uniform(0.4, 0.7)
        }
        
        # Mutation parameters
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 1
        self.mutation_history = []
        
        # Fixed genetic sequence (for pattern matching and inheritance)
        self.genetic_code = self._generate_genetic_code()
    
    def _generate_genetic_code(self) -> str:
        """Generate a DNA-like sequence encoding traits"""
        # Convert traits to a binary-like string representation
        gene = ""
        for trait, value in self.traits.items():
            # Convert value to binary-like string segment
            binary_val = format(int(value * 255), '08b')
            gene += binary_val
            
        return gene
    
    def mutate(self) -> None:
        """Apply random mutations to traits"""
        mutated_traits = {}
        mutation_record = []
        
        for trait, value in self.traits.items():
            # Apply mutation with probability based on mutation rate
            if np.random.random() < self.mutation_rate:
                # Mutation strength decreases with generation
                strength = 0.1 / np.sqrt(self.generation)
                
                # Apply mutation
                mutation = np.random.normal(0, strength)
                new_value = np.clip(value + mutation, 0.1, 0.9)
                
                mutated_traits[trait] = new_value
                mutation_record.append((trait, value, new_value))
            else:
                mutated_traits[trait] = value
        
        # Update traits and record mutations
        self.traits = mutated_traits
        if mutation_record:
            self.mutation_history.append({
                'generation': self.generation,
                'mutations': mutation_record
            })
            
        # Update generation and genetic code
        self.generation += 1
        self.genetic_code = self._generate_genetic_code()
    
    def crossover(self, other_dna: 'EvolvingNodeDNA') -> 'EvolvingNodeDNA':
        """Create a new DNA by crossing over with another DNA"""
        if np.random.random() > self.crossover_rate:
            # No crossover - return copy of self with possible mutations
            child_dna = EvolvingNodeDNA(
                traits=self.traits.copy(),
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate
            )
            child_dna.mutate()
            return child_dna
        
        # Perform crossover
        child_traits = {}
        
        # For each trait, randomly select from either parent
        for trait in self.traits:
            if np.random.random() < 0.5:
                child_traits[trait] = self.traits[trait]
            else:
                child_traits[trait] = other_dna.traits[trait]
        
        # Create child DNA
        child_dna = EvolvingNodeDNA(
            traits=child_traits,
            mutation_rate=(self.mutation_rate + other_dna.mutation_rate) / 2,
            crossover_rate=(self.crossover_rate + other_dna.crossover_rate) / 2
        )
        
        # Apply mutation to child
        child_dna.mutate()
        
        return child_dna
    
    def genetic_similarity(self, other_dna: 'EvolvingNodeDNA') -> float:
        """Calculate genetic similarity between two DNA structures"""
        # Compare genetic codes
        code1 = self.genetic_code
        code2 = other_dna.genetic_code
        
        # Ensure codes are same length by padding shorter one
        max_len = max(len(code1), len(code2))
        code1 = code1.ljust(max_len, '0')
        code2 = code2.ljust(max_len, '0')
        
        # Count matching bits
        matching_bits = sum(c1 == c2 for c1, c2 in zip(code1, code2))
        
        # Calculate similarity (1.0 = identical, 0.0 = completely different)
        return matching_bits / max_len
    
    def get_trait_influence(self) -> Dict[str, float]:
        """Calculate influence factors for node interactions based on traits"""
        return {
            'energy_transfer': self.traits['energy_efficiency'] * self.traits['connection_strength'],
            'tension_response': self.traits['tension_sensitivity'] * self.traits['adaptability'],
            'quantum_effect': self.traits['quantum_coupling'],
            'memory_effect': self.traits['memory_retention'] * self.traits['learning_rate'],
            'adaptation_rate': self.traits['adaptability'] * self.traits['learning_rate']
        }
