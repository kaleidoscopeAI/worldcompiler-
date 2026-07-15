import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class ConsciousCubeVisualizer:
    """3D visualization of the Quantum Conscious Cube"""
    
    def __init__(self, interface: Optional[ConsciousCubeInterface] = None):
        self.interface = interface
        self.colorscales = {
            'energy': 'Viridis',
            'tension': 'Plasma',
            'consciousness': 'Inferno'
        }
    
    def create_3d_visualization(self, state: Dict[str, Any]) -> go.Figure:
        """Create a 3D visualization of the cube's current state"""
        fig = go.Figure()
        
        # Extract network state
        network = state['network']
        
        # Add nodes
        if 'nodes' in network and network['nodes']:
            node_positions = np.array([node['position'] for node in network['nodes']])
            node_energies = np.array([node['energy'] for node in network['nodes']])
            np.array([node.get('local_tension', 0) for node in network['nodes']])
            
            # Scale node sizes by energy
            node_sizes = 10 + 30 * node_energies
            
            # Text labels
            node_text = [f"Node {node['id']}<br>Energy: {node['energy']:.2f}<br>Tension: {node.get('local_tension', 0):.2f}"
                         for node in network['nodes']]
            
            # Plot nodes
            fig.add_trace(go.Scatter3d(
                x=node_positions[:, 0],
                y=node_positions[:, 1],
                z=node_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_energies,
                    colorscale=self.colorscales['energy'],
                    opacity=0.8,
                    colorbar=dict(
                        title="Energy",
                        x=0.9
                    )
                ),
                text=node_text,
                hoverinfo='text',
                name='Nodes'
            ))
        
        # Add connections (edges)
        if 'edges' in network and network['edges']:
            # Create lines for each edge
            edge_x, edge_y, edge_z = [], [], []
            edge_tensions = []
            
            for edge in network['edges']:
                source_idx = next(i for i, node in enumerate(network['nodes']) if node['id'] == edge['source'])
                target_idx = next(i for i, node in enumerate(network['nodes']) if node['id'] == edge['target'])
                
                source_pos = node_positions[source_idx]
                target_pos = node_positions[target_idx]
                
                # Add line coordinates with None to create separation between lines
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
                edge_z.extend([source_pos[2], target_pos[2], None])
                
                # Add tension value twice plus None to match line coordinates
                tension = edge['tension']
                edge_tensions.extend([tension, tension, None])
            
            # Plot edges
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(
                    color=edge_tensions,
                    colorscale=self.colorscales['tension'],
                    width=3
                ),
                opacity=0.5,
                name='Connections'
            ))
        
        # Add high tension points
        if 'high_tension_points' in network and network['high_tension_points']:
            tension_positions = np.array([point['position'] for point in network['high_tension_points']])
            tension_values = np.array([point['tension'] for point in network['high_tension_points']])
            
            # Scale based on tension
            tension_sizes = 3 + 7 * tension_values
            
            fig.add_trace(go.Scatter3d(
                x=tension_positions[:, 0],
                y=tension_positions[:, 1],
                z=tension_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=tension_sizes,
                    color=tension_values,
                    colorscale=self.colorscales['tension'],
                    opacity=0.3,
                    symbol='diamond'
                ),
                name='Tension Field'
            ))
        
        # Set layout
        consciousness_level = state['consciousness']['global_level']
        properties = state['consciousness']['emergent_properties']
        
        fig.update_layout(
            title=f"Quantum Consciousness Cube (Level: {consciousness_level:.3f})",
            scene=dict(
                xaxis=dict(range=[-1.2, 1.2], title='X'),
                yaxis=dict(range=[-1.2, 1.2], title='Y'),
                zaxis=dict(range=[-1.2, 1.2], title='Z'),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            annotations=[
                dict(
                    x=0.01,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f"Harmony: {properties['harmony']:.2f}<br>"
                         f"Complexity: {properties['complexity']:.2f}<br>"
                         f"Self-Organization: {properties['self_organization']:.2f}<br>"
                         f"Adaptability: {properties['adaptability']:.2f}<br>"
                         f"Integration: {properties['integration']:.2f}",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="white"
                    ),
                    align="left",
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=2,
                    borderpad=4
                )
            ]
        )
        
        return fig
    
    def create_molecular_visualization(self, molecule_data: Dict[str, Any]) -> go.Figure:
        """Create a 3D visualization of a molecule"""
        fig = go.Figure()
        
        atoms = molecule_data.get('atoms', [])
        
        # Element colors
        element_colors = {
            'H': 'white',
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'P': 'orange',
            'S': 'yellow'
        }
        
        # Add atoms
        atom_x = []
        atom_y = []
        atom_z = []
        atom_colors = []
        atom_sizes = []
        atom_texts = []
        
        for i, atom in enumerate(atoms):
            pos = atom.get('position', [0, 0, 0])
            atom_x.append(pos[0])
            atom_y.append(pos[1])
            atom_z.append(pos[2])
            
            element = atom.get('element', 'C')
            atom_colors.append(element_colors.get(element, 'gray'))
            
            # Scale atom size by radius
            radius = atom.get('radius', 1.0)
            atom_sizes.append(radius * 10)
            
            atom_texts.append(f"{element} ({i})<br>Charge: {atom.get('charge', 0):.2f}")
        
        # Add atom markers
        fig.add_trace(go.Scatter3d(
            x=atom_x,
            y=atom_y,
            z=atom_z,
            mode='markers',
            marker=dict(
                size=atom_sizes,
                color=atom_colors,
                opacity=0.8
            ),
            text=atom_texts,
            hoverinfo='text',
            name='Atoms'
        ))
        
        # Add bonds
        bond_x = []
        bond_y = []
        bond_z = []
        
        for i, atom in enumerate(atoms):
            bonds = atom.get('bonds', [])
            for bond_idx in bonds:
                if bond_idx > i:  # Only add bond once
                    start_pos = atom.get('position', [0, 0, 0])
                    end_pos = atoms[bond_idx].get('position', [0, 0, 0])
                    
                    bond_x.extend([start_pos[0], end_pos[0], None])
                    bond_y.extend([start_pos[1], end_pos[1], None])
                    bond_z.extend([start_pos[2], end_pos[2], None])
        
        # Add bond lines
        fig.add_trace(go.Scatter3d(
            x=bond_x,
            y=bond_y,
            z=bond_z,
            mode='lines',
            line=dict(
                color='gray',
                width=4
            ),
            hoverinfo='none',
            name='Bonds'
        ))
        
        # Set layout
        fig.update_layout(
            title=f"Molecular Structure: {molecule_data.get('name', 'Molecule')}",
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def create_dashboard(self):
        """Create a Dash dashboard for interactive visualization"""
        # Initialize Dash app
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1("Quantum Consciousness Cube"),
            
            html.Div([
                html.Button('Start Simulation', id='start-button', n_clicks=0),
                html.Button('Stop Simulation', id='stop-button', n_clicks=0),
                html.Button('Evolve Nodes', id='evolve-button', n_clicks=0),
                html.Div(id='status-display', style={'marginTop': '10px'})
            ], style={'marginBottom': '20px'}),
            
            dcc.Tabs([
                dcc.Tab(label='3D Visualization', children=[
                    dcc.Graph(id='cube-3d-graph', style={'height': '800px'}),
                    dcc.Interval(id='cube-interval', interval=1000, n_intervals=0)
                ]),
                dcc.Tab(label='Consciousness Level', children=[
                    dcc.Graph(id='consciousness-graph'),
                    dcc.Interval(id='consciousness-interval', interval=2000, n_intervals=0)
                ]),
                dcc.Tab(label='Node Performance', children=[
                    dcc.Graph(id='node-performance-graph'),
                    dcc.Interval(id='node-interval', interval=2000, n_intervals=0)
                ]),
                dcc.Tab(label='Molecular Binding', children=[
                    html.Div([
                        html.Button('Run Binding Simulation', id='binding-button', n_clicks=0),
                        html.Div(id='binding-status', style={'marginTop': '10px'}),
                        dcc.Graph(id='molecule-graph', style={'height': '600px'})
                    ])
                ])
            ]),
            
            # Hidden divs for storing state
            html.Div(id='simulation-running', style={'display': 'none'}, children='false'),
            html.Div(id='current-state', style={'display': 'none'})
        ])
        
        # Define callbacks
        @app.callback(
            [Output('simulation-running', 'children'),
             Output('status-display', 'children')],
            [Input('start-button', 'n_clicks'),
             Input('stop-button', 'n_clicks')],
            [dash.dependencies.State('simulation-running', 'children')]
        )
        def toggle_simulation(start_clicks, stop_clicks, simulation_running):
            ctx = dash.callback_context
            if not ctx.triggered:
                # No button clicked yet
                return 'false', "Simulation not running"
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-button' and simulation_running == 'false':
                return 'true', "Simulation running"
            elif button_id == 'stop-button' and simulation_running == 'true':
                return 'false', "Simulation stopped"
            
            # No change
            return simulation_running, "Simulation running" if simulation_running == 'true' else "Simulation not running"
        
        @app.callback(
            Output('current-state', 'children'),
            [Input('cube-interval', 'n_intervals')],
            [dash.dependencies.State('simulation-running', 'children')]
        )
        def update_simulation(n_intervals, simulation_running):
            if simulation_running == 'true' and self.interface is not None:
                # Run simulation step
                self.interface.simulate_step()
                
                # Get current state
                state = self.interface.get_state()
                
                # Convert to JSON
                import json
                return json.dumps(state)
            
            return dash.no_update
        
        @app.callback(
            Output('cube-3d-graph', 'figure'),
            [Input('current-state', 'children')]
        )
        def update_3d_visualization(state_json):
            if state_json is None:
                # Create empty visualization
                fig = go.Figure()
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-1.2, 1.2]),
                        yaxis=dict(range=[-1.2, 1.2]),
                        zaxis=dict(range=[-1.2, 1.2]),
                        aspectmode='cube'
                    ),
                    title="No data available"
                )
                return fig
                
            # Parse state
            import json
            state = json.loads(state_json)
            
            # Create visualization
            return self.create_3d_visualization(state)
        
        @app.callback(
            Output('consciousness-graph', 'figure'),
            [Input('consciousness-interval', 'n_intervals')],
            [dash.dependencies.State('current-state', 'children')]
        )
        def update_consciousness_graph(n_intervals, state_json):
            fig = go.Figure()
            
            if state_json is not None:
                # Parse state
                import json
                state = json.loads(state_json)
                
                # Extract consciousness history
                history = state['stats'].get('consciousness_history', [])
                
                if history:
                    # Create line graph
                    x = list(range(len(history)))
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=history,
                        mode='lines',
                        line=dict(width=2, color='purple'),
                        name='Consciousness Level'
                    ))
                    
                    # Add threshold line
                    if self.interface:
                        threshold = self.interface.consciousness_threshold
                        fig.add_trace(go.Scatter(
                            x=[0, len(history)-1],
                            y=[threshold, threshold],
                            mode='lines',
                            line=dict(width=1, color='red', dash='dash'),
                            name='Threshold'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Consciousness Level Evolution",
                        xaxis=dict(title='Simulation Step'),
                        yaxis=dict(title='Consciousness Level', range=[0, 1]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
            
            if not fig.data:
                fig.update_layout(
                    title="No consciousness data available"
                )
            
            return fig
        
        @app.callback(
            Output('node-performance-graph', 'figure'),
            [Input('node-interval', 'n_intervals')],
            [dash.dependencies.State('current-state', 'children')]
        )
        def update_node_performance(n_intervals, state_json):
            fig = go.Figure()
            
            if state_json is not None:
                # Parse state
                import json
                json.loads(state_json)
                
                if self.interface and self.interface.nodes:
                    # Extract node performance data
                    node_ids = []
                    performances = []
                    energies = []
                    
                    for node_id, node_data in self.interface.nodes.items():
                        node_ids.append(node_id)
                        performances.append(node_data['performance'])
                        energies.append(node_data['properties'].get('energy', 0))
                    
                    # Sort by performance
                    sorted_indices = np.argsort(performances)[::-1]  # Reverse for descending
                    
                    node_ids = [node_ids[i] for i in sorted_indices]
                    performances = [performances[i] for i in sorted_indices]
                    energies = [energies[i] for i in sorted_indices]
                    
                    # Create bar chart
                    fig.add_trace(go.Bar(
                        x=node_ids,
                        y=performances,
                        name='Performance',
                        marker_color='blue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=node_ids,
                        y=energies,
                        name='Energy',
                        marker_color='orange'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Node Performance",
                        xaxis=dict(title='Node ID'),
                        yaxis=dict(title='Value', range=[0, 1]),
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
            
            if not fig.data:
                fig.update_layout(
                    title="No node performance data available"
                )
            
            return fig
        
        @app.callback(
            Output('status-display', 'children'),
            [Input('evolve-button', 'n_clicks')],
            [dash.dependencies.State('simulation-running', 'children')]
        )
        def evolve_nodes(n_clicks, simulation_running):
            if n_clicks == 0:
                return dash.no_update
                
            if self.interface:
                self.interface.evolve_nodes()
                return "Evolution cycle completed"
            
            return "No interface available for evolution"
        
        return app
