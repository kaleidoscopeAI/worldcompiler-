#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED ADVANCED COMPUTATIONAL SYSTEM
======================================

Real merge of two of the three components you asked for:

  1. advanced_computational_system_base.py — Graph / Numerical / ML / System
     tabs (your original "Groundbreaking Computational System").
  2. relational_epistemic_substrate.py — RBCube lattice, Hamiltonian dynamics,
     HypothesisRegistry, splits/merges — added here as a fifth "Relational
     Substrate" tab with a live 3D matplotlib view.

NOT merged, and why:
  The third piece from your original merge stub — OrganicAISystem /
  OrganicWebSocketServer, feeding tensioncube_.html over a live WebSocket —
  was never provided as actual Python source, only as a description. I
  checked tensioncube_.html directly: it has no WebSocket client code at
  all: the cube simulation and the "Ollama" chat both run entirely in
  browser-side JavaScript with a setTimeout()-faked response. There is
  nothing here to merge for that piece. If you want a real backend driving
  that HTML file, that's new code to write, not a merge of anything that
  currently exists.

Run:
    python3 advanced_unified_system.py

Dependencies beyond the base file: none new (relational_epistemic_substrate
already only needs numpy/scipy, both already required by the base system).
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import matplotlib.pyplot as plt

from advanced_computational_system_base import AdvancedComputationalSystem
from relational_epistemic_substrate import RBNetwork, RBCube, HypothesisRegistry


# ---------------------------------------------------------------------------
# Face-local 3D layout helper
# ---------------------------------------------------------------------------
# Maps each RBNode's (face, u, v) lattice coordinate onto the surface of a
# unit cube in 3D, so the "Relational Substrate" tab can show something
# geometrically meaningful rather than an arbitrary force-directed blob.

_FACE_NORMALS = {
    0: (0.0, None, None),   # Left    (x = 0)
    1: (1.0, None, None),   # Right   (x = 1)
    2: (None, 0.0, None),   # Front   (y = 0)
    3: (None, 1.0, None),   # Back    (y = 1)
    4: (None, None, 1.0),   # Top     (z = 1)
    5: (None, None, 0.0),   # Bottom  (z = 0)
}


def node_xyz(node):
    """Project an RBNode's (face, u, v) onto a unit cube surface."""
    fx, fy, fz = _FACE_NORMALS.get(node.face, (0.0, 0.0, 0.0))
    u, v = node.u, node.v
    x = fx if fx is not None else u
    y = fy if fy is not None else (u if fx is not None else v)
    z = fz if fz is not None else v
    # For side faces (fx or fy fixed), use (u, v) for the other two axes
    if node.face in (0, 1):
        x = fx
        y = u
        z = v
    elif node.face in (2, 3):
        x = u
        y = fy
        z = v
    elif node.face in (4, 5):
        x = u
        y = v
        z = fz
    return x, y, z


class UnifiedSystem(AdvancedComputationalSystem):
    """AdvancedComputationalSystem + a fifth Relational Substrate tab."""

    def __init__(self):
        # build_gui=False: create the c_lib/asm_lib/graph/numerical/ml
        # components, but hold off on building the window so we can set up
        # relational-substrate state before any tab is constructed.
        super().__init__(build_gui=False)

        # Relational substrate state (created lazily on "Start")
        self.rel_net = None
        self.rel_running = False
        self.rel_thread = None
        self.rel_queue = queue.Queue()
        self.rel_tick = 0

        self._build_gui()
        self._poll_rel_queue()

    # ------------------------------------------------------------------
    # Hook provided by the base class
    # ------------------------------------------------------------------

    def create_extra_tabs(self):
        self.create_relational_tab()

    # ------------------------------------------------------------------
    # Relational Substrate tab
    # ------------------------------------------------------------------

    def create_relational_tab(self):
        """Create the Relational Epistemic Substrate tab."""
        rel_frame = ttk.Frame(self.notebook)
        self.notebook.add(rel_frame, text="Relational Substrate")

        # Left panel: controls
        ctrl_frame = ttk.LabelFrame(rel_frame, text="Cube Controls")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(ctrl_frame, text="Start Simulation", command=self.start_rel_sim)\
            .grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.EW)
        ttk.Button(ctrl_frame, text="Stop Simulation", command=self.stop_rel_sim)\
            .grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.EW)
        ttk.Button(ctrl_frame, text="Force Split (Cube 0)", command=self.force_split)\
            .grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)
        ttk.Button(ctrl_frame, text="Import Registry -> Graph Tab", command=self.import_registry_to_graph)\
            .grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)

        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)

        ttk.Label(ctrl_frame, text="Live stats:").grid(row=5, column=0, columnspan=2, sticky=tk.W)
        self.rel_stats_var = tk.StringVar(value="Not started")
        ttk.Label(ctrl_frame, textvariable=self.rel_stats_var, justify=tk.LEFT)\
            .grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Right: 3D view + log
        result_frame = ttk.LabelFrame(rel_frame, text="Cube 0 — Lattice View")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.rel_canvas_frame = ttk.Frame(result_frame)
        self.rel_canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.rel_log_text = tk.Text(result_frame, height=8, wrap=tk.WORD)
        self.rel_log_text.pack(fill=tk.X, expand=False, pady=10)

        self.rel_log_message("Relational Substrate tab ready. Press Start to seed and run RBNetwork.")

    def rel_log_message(self, msg):
        self.rel_log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.rel_log_text.see(tk.END)

    # ------------------------------------------------------------------
    # Simulation control (background thread; Tk touched only via queue)
    # ------------------------------------------------------------------

    def start_rel_sim(self):
        if self.rel_running:
            messagebox.showinfo("Info", "Simulation already running")
            return
        if self.rel_net is None:
            self.rel_net = RBNetwork()
            self.rel_net.seed()
            self.rel_tick = 0
            self.rel_log_message("RBNetwork created and seeded (300-node lattice + KRAS_G12D fragment nodes).")

        self.rel_running = True
        self.rel_thread = threading.Thread(target=self._rel_sim_loop, daemon=True)
        self.rel_thread.start()
        self.rel_log_message("Simulation started.")

    def stop_rel_sim(self):
        self.rel_running = False
        self.rel_log_message("Simulation stop requested.")

    def _rel_sim_loop(self):
        """Runs in a background thread. Never touches Tk widgets directly —
        pushes snapshots onto rel_queue for the main thread to consume."""
        while self.rel_running:
            self.rel_net.step(self.rel_tick)
            self.rel_tick += 1

            if self.rel_tick % 5 == 0:
                snapshot = {
                    "tick": self.rel_tick,
                    "num_cubes": len(self.rel_net.cubes),
                    "registry_keys": len(self.rel_net.registry.entries),
                    "cube0_summary": self.rel_net.cubes[0].summary() if self.rel_net.cubes else "no cubes",
                    "cube0_nodes": list(self.rel_net.cubes[0].nodes.values()) if self.rel_net.cubes else [],
                }
                self.rel_queue.put(snapshot)

            time.sleep(0.02)

    def _poll_rel_queue(self):
        """Runs on the main Tk thread via root.after; drains the queue and
        updates widgets safely."""
        latest = None
        try:
            while True:
                latest = self.rel_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._update_rel_gui(latest)

        # Reschedule regardless of whether the relational tab has been
        # built yet or a simulation is running.
        self.root.after(200, self._poll_rel_queue)

    def _update_rel_gui(self, snapshot):
        if hasattr(self, "rel_stats_var"):
            self.rel_stats_var.set(
                f"Tick: {snapshot['tick']}\n"
                f"Cubes: {snapshot['num_cubes']}\n"
                f"Registry keys: {snapshot['registry_keys']}"
            )
        if hasattr(self, "rel_canvas_frame"):
            self._visualize_cube(snapshot["cube0_nodes"])

    def _visualize_cube(self, nodes):
        if not nodes:
            return
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        xs, ys, zs, colors = [], [], [], []
        for n in nodes:
            x, y, z = node_xyz(n)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            colors.append(n.rstate.er_bridge_strength())

        sc = ax.scatter(xs, ys, zs, c=colors, cmap="plasma", s=12)
        ax.set_title("Cube 0 boundary nodes (colored by ER-bridge strength)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="bridge strength")

        for widget in self.rel_canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, self.rel_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)  # avoid leaking figures across ticks

    # ------------------------------------------------------------------
    # Extra actions
    # ------------------------------------------------------------------

    def force_split(self):
        if self.rel_net is None or not self.rel_net.cubes:
            messagebox.showinfo("Info", "Start the simulation first")
            return
        cube = self.rel_net.cubes[0]
        try:
            child = cube.split(self.rel_net._next_id)
            self.rel_net._next_id += 1
            self.rel_net.cubes.append(child)
            self.rel_log_message(
                f"Forced split of cube {cube.id}: child cube {child.id} created "
                f"with {len(child.nodes)} nodes; parent retains {len(cube.nodes)}."
            )
        except Exception as e:
            self.rel_log_message(f"Split failed: {e}")

    def import_registry_to_graph(self):
        """Pull HypothesisRegistry entries into the Graph Theory tab as a
        NetworkX graph: one node per registry key, edges to parents."""
        if self.rel_net is None or not self.rel_net.registry.entries:
            messagebox.showinfo("Info", "No registry entries yet — start the simulation first")
            return

        import networkx as nx
        g = nx.DiGraph()
        for key, entry in self.rel_net.registry.entries.items():
            g.add_node(key, budget=entry.budget, status=entry.status)
            for parent in entry.parents:
                g.add_edge(parent, key)

        # Clear stale results from any previously loaded graph (e.g.
        # detect_communities' 'communities' mapping keyed on old node ids) --
        # visualize_graph() would otherwise KeyError looking up new node
        # labels in an old mapping.
        self.graph_processor.results = {}
        self.graph_processor.graph = g
        self.graph_result_text.delete(1.0, tk.END)
        self.graph_result_text.insert(
            tk.END,
            f"Imported {g.number_of_nodes()} registry entries as graph nodes "
            f"({g.number_of_edges()} parent/child edges).\n"
        )
        self.rel_log_message(f"Imported {g.number_of_nodes()} registry entries into Graph Theory tab.")
        try:
            self.visualize_graph()
        except Exception:
            pass  # visualize_graph shows its own error dialog on failure


if __name__ == "__main__":
    app = UnifiedSystem()
    app.run()
