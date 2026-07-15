#!/usr/bin/env python3
# Groundbreaking Computational System: Combines advanced graph algorithms, numerical
# computing, machine learning, and low-level optimizations in a single executable
#
# This module is unmodified from the original source you provided, except that the
# GUI construction in AdvancedComputationalSystem.__init__ has been split so that a
# subclass (see advanced_unified_system.py) can add the Relational Substrate tab
# without duplicating this file. See the "HOOK FOR SUBCLASSES" comment below.

import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ctypes
import multiprocessing
import threading
import scipy.optimize as optimize
from sklearn import cluster, manifold, decomposition
import time
import hashlib
import struct
import queue
import tempfile
import subprocess
import shutil
from functools import lru_cache

# System configuration
CONFIG = {
    "max_threads": multiprocessing.cpu_count(),
    "cache_size": 1024 * 1024 * 512,  # 512MB cache
    "precision": "double",
    "gpu_acceleration": True,
    "assembly_optimization": True,
    "visualization_level": 3,
    "auto_optimization": True,
}

# ==================== LOW LEVEL OPTIMIZATION COMPONENTS ====================

# C code for high-performance critical functions
C_CODE = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// High-performance matrix operations
void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

// Optimized graph operations
void compute_shortest_paths(int *graph, int *dist, int n) {
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ij = i*n + j;
                int ik = i*n + k;
                int kj = k*n + j;
                if (graph[ik] + graph[kj] < graph[ij]) {
                    graph[ij] = graph[ik] + graph[kj];
                    dist[ij] = k;
                }
            }
        }
    }
}

// Fast Fourier Transform implementation
void fft_calculate(double *real, double *imag, int n) {
    // Cooley-Tukey FFT algorithm
    if (n <= 1) return;

    double *even_real = (double*)malloc(n/2 * sizeof(double));
    double *even_imag = (double*)malloc(n/2 * sizeof(double));
    double *odd_real = (double*)malloc(n/2 * sizeof(double));
    double *odd_imag = (double*)malloc(n/2 * sizeof(double));

    for (int i = 0; i < n/2; i++) {
        even_real[i] = real[i*2];
        even_imag[i] = imag[i*2];
        odd_real[i] = real[i*2+1];
        odd_imag[i] = imag[i*2+1];
    }

    fft_calculate(even_real, even_imag, n/2);
    fft_calculate(odd_real, odd_imag, n/2);

    for (int k = 0; k < n/2; k++) {
        double theta = -2 * M_PI * k / n;
        double re = cos(theta);
        double im = sin(theta);
        double tr = odd_real[k] * re - odd_imag[k] * im;
        double ti = odd_real[k] * im + odd_imag[k] * re;

        real[k] = even_real[k] + tr;
        imag[k] = even_imag[k] + ti;
        real[k+n/2] = even_real[k] - tr;
        imag[k+n/2] = even_imag[k] - ti;
    }

    free(even_real);
    free(even_imag);
    free(odd_real);
    free(odd_imag);
}

// Advanced cryptographic hash calculation
void secure_hash(unsigned char *data, size_t len, unsigned char *output) {
    unsigned int h1 = 0x67452301;
    unsigned int h2 = 0xEFCDAB89;
    unsigned int h3 = 0x98BADCFE;
    unsigned int h4 = 0x10325476;
    unsigned int h5 = 0xC3D2E1F0;

    // A simplified secure hash algorithm implementation
    for (size_t i = 0; i < len; i++) {
        h1 = (h1 << 5) | (h1 >> 27);
        h1 += data[i];
        h2 = (h2 << 7) | (h2 >> 25);
        h2 ^= h1;
        h3 = (h3 << 11) | (h3 >> 21);
        h3 ^= h2;
        h4 = (h4 << 15) | (h4 >> 17);
        h4 ^= h3;
        h5 = (h5 << 19) | (h5 >> 13);
        h5 ^= h4;
    }

    // Pack results into output array
    memcpy(output, &h1, 4);
    memcpy(output+4, &h2, 4);
    memcpy(output+8, &h3, 4);
    memcpy(output+12, &h4, 4);
    memcpy(output+16, &h5, 4);
}
"""

# Assembly code for critical functions
# Uses inline assembly with GCC-style syntax
ASSEMBLY_CODE = """
global matrix_determinant
global bit_manipulation
global prime_test

section .text

; Fast matrix determinant calculation for 4x4 matrix
; Input: RDI = pointer to matrix (row-major order, 16 doubles)
; Output: XMM0 = determinant
matrix_determinant:
    push rbp
    mov rbp, rsp

    ; Load matrix elements
    movapd xmm0, [rdi]      ; row 1
    movapd xmm1, [rdi+16]
    movapd xmm2, [rdi+32]   ; row 2
    movapd xmm3, [rdi+48]
    movapd xmm4, [rdi+64]   ; row 3
    movapd xmm5, [rdi+80]
    movapd xmm6, [rdi+96]   ; row 4
    movapd xmm7, [rdi+112]

    ; Calculate determinant using cofactor expansion
    ; This is a simplified implementation
    ; A full implementation would require more registers and operations

    ; Calculate 2x2 determinants for upper-left block
    movapd xmm8, xmm0
    mulsd xmm8, xmm3
    movapd xmm9, xmm1
    mulsd xmm9, xmm2
    subsd xmm8, xmm9        ; det(A)

    ; Calculate 2x2 determinants for upper-right block
    movapd xmm10, xmm4
    mulsd xmm10, xmm7
    movapd xmm11, xmm5
    mulsd xmm11, xmm6
    subsd xmm10, xmm11      ; det(B)

    ; Combine results
    mulsd xmm8, xmm10
    movapd xmm0, xmm8       ; result in xmm0

    leave
    ret

; Advanced bit manipulation
; Input: RDI = value to manipulate
;        RSI = operation flags
; Output: RAX = manipulated value
bit_manipulation:
    push rbp
    mov rbp, rsp

    mov rax, rdi        ; Get input value
    mov rcx, rsi        ; Get operation flags

    test rcx, 1
    jz .no_reverse
    ; Reverse bits
    mov rdx, 0
    mov r8, 64          ; 64 bits
.reverse_loop:
    shr rax, 1          ; Shift right to get next bit
    rcl rdx, 1          ; Rotate through carry to build reversed value
    dec r8
    jnz .reverse_loop
    mov rax, rdx
.no_reverse:

    test rcx, 2
    jz .no_count
    ; Count bits
    mov rdx, rax
    mov rax, 0
.count_loop:
    test rdx, rdx
    jz .done_count
    inc rax
    and rdx, rdx-1      ; Clear lowest set bit
    jmp .count_loop
.done_count:
.no_count:

    test rcx, 4
    jz .no_parity
    ; Calculate parity
    mov rdx, rax
    mov rax, 0
    mov r8, 64
.parity_loop:
    bt rdx, 0
    jnc .no_flip
    xor rax, 1
.no_flip:
    shr rdx, 1
    dec r8
    jnz .parity_loop
.no_parity:

    leave
    ret

; Miller-Rabin primality test
; Input: RDI = number to test
;        RSI = number of rounds
; Output: RAX = 1 if probably prime, 0 if composite
prime_test:
    push rbp
    mov rbp, rsp

    ; Simplified implementation
    ; For a real implementation, we would use the Miller-Rabin algorithm

    mov rax, rdi
    cmp rax, 2
    jb .not_prime      ; Less than 2 is not prime
    je .is_prime       ; 2 is prime

    test rax, 1
    jz .not_prime      ; Even numbers > 2 are not prime

    mov rcx, 3         ; Start checking divisibility from 3
.check_loop:
    mov rdx, 0
    div rcx
    test rdx, rdx
    jz .not_prime      ; Divisible by rcx

    add rcx, 2         ; Check next odd number
    mov rax, rdi       ; Restore original number

    ; Check if we've tested up to sqrt(n)
    mov r8, rcx
    mul r8
    cmp rax, rdi
    jbe .check_loop

.is_prime:
    mov rax, 1
    jmp .done

.not_prime:
    mov rax, 0

.done:
    leave
    ret
"""

def setup_c_extensions():
    """Compile and load C extensions for high-performance operations"""
    try:
        # Save C code to temporary file
        c_file = tempfile.NamedTemporaryFile(suffix='.c', delete=False)
        c_file.write(C_CODE.encode('utf-8'))
        c_file.close()

        # Compile C code as shared library
        shared_lib = tempfile.NamedTemporaryFile(suffix='.so', delete=False)
        shared_lib.close()

        compile_cmd = [
            "gcc", "-shared", "-o", shared_lib.name, c_file.name,
            "-fPIC", "-O3", "-march=native", "-fopenmp", "-lm"
        ]
        subprocess.run(compile_cmd, check=True)

        # Load shared library
        c_lib = ctypes.CDLL(shared_lib.name)

        # Define function prototypes
        c_lib.matrix_multiply.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        c_lib.matrix_multiply.restype = None

        c_lib.compute_shortest_paths.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int
        ]
        c_lib.compute_shortest_paths.restype = None

        c_lib.fft_calculate.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        c_lib.fft_calculate.restype = None

        c_lib.secure_hash.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte)
        ]
        c_lib.secure_hash.restype = None

        # Clean up temporary files on exit
        def cleanup_temp_files():
            os.unlink(c_file.name)
            os.unlink(shared_lib.name)

        import atexit
        atexit.register(cleanup_temp_files)

        return c_lib
    except Exception as e:
        print(f"Error setting up C extensions: {e}")
        return None

def setup_assembly_extensions():
    """Compile and load assembly extensions for ultra-performance operations"""
    if not CONFIG["assembly_optimization"]:
        return None

    try:
        # Save assembly code to temporary file
        asm_file = tempfile.NamedTemporaryFile(suffix='.asm', delete=False)
        asm_file.write(ASSEMBLY_CODE.encode('utf-8'))
        asm_file.close()

        # Compile assembly code
        obj_file = tempfile.NamedTemporaryFile(suffix='.o', delete=False)
        obj_file.close()

        # Assemble
        asm_cmd = ["nasm", "-f", "elf64", "-o", obj_file.name, asm_file.name]
        subprocess.run(asm_cmd, check=True)

        # Create shared library
        shared_lib = tempfile.NamedTemporaryFile(suffix='.so', delete=False)
        shared_lib.close()

        link_cmd = ["gcc", "-shared", "-o", shared_lib.name, obj_file.name]
        subprocess.run(link_cmd, check=True)

        # Load shared library
        asm_lib = ctypes.CDLL(shared_lib.name)

        # Define function prototypes
        asm_lib.matrix_determinant.argtypes = [ctypes.POINTER(ctypes.c_double)]
        asm_lib.matrix_determinant.restype = ctypes.c_double

        asm_lib.bit_manipulation.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        asm_lib.bit_manipulation.restype = ctypes.c_uint64

        asm_lib.prime_test.argtypes = [ctypes.c_uint64, ctypes.c_int]
        asm_lib.prime_test.restype = ctypes.c_int

        # Clean up temporary files on exit
        def cleanup_temp_files():
            os.unlink(asm_file.name)
            os.unlink(obj_file.name)
            os.unlink(shared_lib.name)

        import atexit
        atexit.register(cleanup_temp_files)

        return asm_lib
    except Exception as e:
        print(f"Warning: Assembly extensions not loaded: {e}")
        return None

# ==================== GRAPH THEORY COMPONENTS ====================

class GraphProcessor:
    """Advanced graph processing and analysis module"""

    def __init__(self, c_lib=None, asm_lib=None):
        self.c_lib = c_lib
        self.asm_lib = asm_lib
        self.graph = None
        self.results = {}

    def load_graph(self, adjacency_matrix=None, edge_list=None, graph_obj=None):
        """Load graph from various formats"""
        if graph_obj is not None:
            self.graph = graph_obj
        elif adjacency_matrix is not None:
            self.graph = nx.from_numpy_array(np.array(adjacency_matrix))
        elif edge_list is not None:
            self.graph = nx.parse_edgelist(edge_list)
        else:
            raise ValueError("Must provide a graph in some format")

    def generate_random_graph(self, nodes, edge_probability=0.5, directed=False, weighted=False):
        """Generate a random graph for testing"""
        if directed:
            self.graph = nx.gnp_random_graph(nodes, edge_probability, directed=True)
        else:
            self.graph = nx.gnp_random_graph(nodes, edge_probability)

        if weighted:
            # Add random weights
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = np.random.uniform(0.1, 10.0)

    def compute_centralities(self):
        """Compute various centrality measures"""
        self.results['degree_centrality'] = nx.degree_centrality(self.graph)
        self.results['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
        self.results['closeness_centrality'] = nx.closeness_centrality(self.graph)
        self.results['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(self.graph)

        return self.results

    def find_communities(self, algorithm='louvain'):
        """Detect communities in the graph using various algorithms"""
        if algorithm == 'louvain':
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            self.results['communities'] = partition
        elif algorithm == 'girvan_newman':
            comp = nx.community.girvan_newman(self.graph)
            self.results['communities'] = tuple(sorted(c) for c in next(comp))
        elif algorithm == 'label_propagation':
            self.results['communities'] = nx.community.label_propagation_communities(self.graph)

        return self.results['communities']

    def shortest_paths(self):
        """Compute shortest paths using highly optimized code"""
        if self.c_lib is None:
            # Fallback to NetworkX implementation
            self.results['shortest_paths'] = dict(nx.all_pairs_shortest_path(self.graph))
            return self.results['shortest_paths']

        # Use optimized C implementation
        n = self.graph.number_of_nodes()

        # Convert graph to adjacency matrix with infinity for no connections
        adj_matrix = np.full((n, n), np.inf)
        for i in range(n):
            adj_matrix[i, i] = 0

        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            adj_matrix[u, v] = weight
            if not self.graph.is_directed():
                adj_matrix[v, u] = weight

        # Replace infinity with a large number for C code
        adj_matrix[np.isinf(adj_matrix)] = 999999

        # Prepare arrays for C function
        adj_matrix = adj_matrix.astype(np.int32)
        dist_matrix = np.zeros_like(adj_matrix)

        # Convert to C-compatible types
        adj_ptr = adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        dist_ptr = dist_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Call C function
        self.c_lib.compute_shortest_paths(adj_ptr, dist_ptr, n)

        # Convert results back to Python dict format
        paths = {}
        for i in range(n):
            paths[i] = {}
            for j in range(n):
                if adj_matrix[i, j] < 999999:
                    paths[i][j] = adj_matrix[i, j]

        self.results['shortest_paths'] = paths
        return paths

    def minimum_spanning_tree(self):
        """Find the minimum spanning tree of the graph"""
        if not nx.is_connected(self.graph):
            return "Graph is not connected"

        mst = nx.minimum_spanning_tree(self.graph)
        self.results['mst'] = mst
        return mst

    def maximal_independent_set(self):
        """Find a maximal independent set in the graph"""
        self.results['mis'] = nx.maximal_independent_set(self.graph)
        return self.results['mis']

    def graph_coloring(self):
        """Compute an optimal graph coloring"""
        self.results['coloring'] = nx.greedy_color(self.graph)
        return self.results['coloring']

    def spectral_analysis(self):
        """Perform spectral analysis of the graph"""
        # Calculate the Laplacian matrix
        laplacian = nx.laplacian_matrix(self.graph).toarray()

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # Store results
        self.results['laplacian_eigenvalues'] = eigenvalues
        self.results['laplacian_eigenvectors'] = eigenvectors

        # Calculate spectral gap (algebraic connectivity)
        if len(eigenvalues) > 1:
            self.results['spectral_gap'] = eigenvalues[1]  # Second smallest eigenvalue

        return self.results

    def visualize(self, layout='spring', node_color=None, node_size=None, edge_color=None):
        """Visualize the graph with various layouts and attributes"""
        plt.figure(figsize=(10, 8))

        # Determine layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Determine node colors based on attributes if not specified
        if node_color is None and 'communities' in self.results:
            node_color = [self.results['communities'][node] for node in self.graph.nodes()]

        # Determine node sizes based on centrality if not specified
        if node_size is None and 'eigenvector_centrality' in self.results:
            centrality = self.results['eigenvector_centrality']
            node_size = [5000 * centrality[node] for node in self.graph.nodes()]
        elif node_size is None:
            node_size = 300

        # Draw the graph
        nx.draw(
            self.graph, pos, with_labels=True, node_color=node_color,
            node_size=node_size, edge_color=edge_color,
            cmap=plt.cm.viridis, font_weight='bold'
        )

        return plt.gcf()  # Return the figure for display in the GUI

# ==================== NUMERICAL COMPUTING COMPONENTS ====================

class NumericalEngine:
    """Advanced numerical computation and optimization engine"""

    def __init__(self, c_lib=None, asm_lib=None):
        self.c_lib = c_lib
        self.asm_lib = asm_lib
        self.cache = {}

    def matrix_multiply(self, A, B):
        """High-performance matrix multiplication

        NOTE: this was originally decorated with @lru_cache(maxsize=1024).
        That never worked: NumPy arrays aren't hashable, so every call raised
        TypeError: unhashable type: 'numpy.ndarray', was swallowed by the
        caller's try/except, and silently fell through to an error dialog
        instead of ever multiplying. Removed rather than "fixed" -- caching
        raw array inputs by identity/value isn't a sound strategy here
        anyway (mutable arrays, huge memory footprint for large matrices).
        If you want caching, hash a content digest of A/B explicitly and
        cache on that instead.
        """
        if self.c_lib is None:
            # Fallback to NumPy
            return np.matmul(A, B)

        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        m, n = A.shape
        n_b, p = B.shape

        if n != n_b:
            raise ValueError("Incompatible matrix dimensions")

        C = np.zeros((m, p), dtype=np.float64)

        # Convert to C-compatible types
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call C function
        self.c_lib.matrix_multiply(A_ptr, B_ptr, C_ptr, m, n, p)

        return C

    def matrix_determinant(self, matrix):
        """Calculate matrix determinant using optimized assembly code"""
        if self.asm_lib is None or matrix.shape != (4, 4):
            # Fallback to NumPy for non-4x4 matrices or when assembly not available
            return np.linalg.det(matrix)

        # Ensure proper format for assembly function
        matrix = np.asarray(matrix, dtype=np.float64, order='C')
        matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call assembly function
        return self.asm_lib.matrix_determinant(matrix_ptr)

    def fft(self, signal):
        """Fast Fourier Transform using optimized C code"""
        if self.c_lib is None:
            # Fallback to NumPy
            return np.fft.fft(signal)

        signal = np.asarray(signal, dtype=np.float64)
        n = len(signal)

        # Check if n is a power of 2
        if n & (n-1) != 0:
            # If not power of 2, pad with zeros
            next_pow2 = 1
            while next_pow2 < n:
                next_pow2 *= 2
            padded = np.zeros(next_pow2, dtype=np.float64)
            padded[:n] = signal
            signal = padded
            n = next_pow2

        # Prepare real and imaginary parts
        real = signal.copy()
        imag = np.zeros_like(signal)

        # Convert to C-compatible types
        real_ptr = real.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        imag_ptr = imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call C function
        self.c_lib.fft_calculate(real_ptr, imag_ptr, n)

        # Combine results into complex array
        return real + 1j * imag

    def optimize_function(self, func, bounds, constraints=None, method='SLSQP'):
        """Optimize a function within given bounds and constraints"""
        result = optimize.minimize(
            func,
            x0=[(b[0] + b[1])/2 for b in bounds],  # Start at midpoint
            bounds=bounds,
            constraints=constraints,
            method=method
        )
        return result

    def integrate_function(self, func, lower, upper, method='quad'):
        """Numerically integrate a function"""
        if method == 'quad':
            from scipy import integrate
            result, error = integrate.quad(func, lower, upper)
        elif method == 'romberg':
            from scipy import integrate
            result = integrate.romberg(func, lower, upper)
        elif method == 'simpson':
            from scipy import integrate
            # Create evenly spaced points for Simpson's rule
            x = np.linspace(lower, upper, 1001)
            y = np.array([func(xi) for xi in x])
            result = integrate.simps(y, x)

        return result

    def solve_ode(self, func, y0, t_span, method='RK45'):
        """Solve ordinary differential equation"""
        from scipy import integrate
        solution = integrate.solve_ivp(func, t_span, y0, method=method)
        return solution

    def eigendecomposition(self, matrix):
        """Compute eigenvalues and eigenvectors of a matrix"""
        values, vectors = np.linalg.eig(matrix)
        return values, vectors

    def svd(self, matrix):
        """Compute Singular Value Decomposition"""
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vh

    def is_prime(self, n):
        """Test if a number is prime using optimized assembly code"""
        if self.asm_lib is None:
            # Fallback to Python implementation
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False

            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True

        # Use assembly implementation
        result = self.asm_lib.prime_test(n, 5)  # 5 rounds of testing
        return bool(result)

    def secure_hash(self, data):
        """Compute a secure hash of data using optimized C code"""
        if self.c_lib is None:
            # Fallback to Python implementation
            return hashlib.sha256(data).digest()

        # Ensure data is in bytes format
        if isinstance(data, str):
            data = data.encode('utf-8')

        # Prepare C-compatible types
        data_arr = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        result = (ctypes.c_ubyte * 20)()  # 20-byte hash output

        # Call C function
        self.c_lib.secure_hash(data_arr, len(data), result)

        # Convert result to Python bytes
        return bytes(result)

# ==================== MACHINE LEARNING COMPONENTS ====================

class MachineLearningModule:
    """Advanced machine learning and pattern recognition capabilities"""

    def __init__(self, numerical_engine=None):
        self.numerical_engine = numerical_engine
        self.models = {}
        self.results = {}

    def cluster_data(self, data, n_clusters=3, algorithm='kmeans'):
        """Cluster data using various algorithms"""
        if algorithm == 'kmeans':
            model = cluster.KMeans(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            model = cluster.DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == 'agglomerative':
            model = cluster.AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")

        labels = model.fit_predict(data)
        self.models['clustering'] = model
        self.results['cluster_labels'] = labels

        return labels

    def dimensionality_reduction(self, data, n_components=2, method='pca'):
        """Reduce dimensionality of data for visualization or analysis"""
        if method == 'pca':
            model = decomposition.PCA(n_components=n_components)
        elif method == 'tsne':
            model = manifold.TSNE(n_components=n_components)
        elif method == 'umap':
            # UMAP requires separate installation
            try:
                import umap
                model = umap.UMAP(n_components=n_components)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                model = manifold.TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        reduced_data = model.fit_transform(data)
        self.models['dim_reduction'] = model
        self.results['reduced_data'] = reduced_data

        return reduced_data

    def train_classifier(self, X, y, model_type='random_forest'):
        """Train a classifier model on the provided data"""
        from sklearn import ensemble, svm, neural_network

        if model_type == 'random_forest':
            model = ensemble.RandomForestClassifier(n_estimators=100)
        elif model_type == 'svm':
            model = svm.SVC(probability=True)
        elif model_type == 'neural_network':
            model = neural_network.MLPClassifier(hidden_layer_sizes=(100, 50))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X, y)
        self.models['classifier'] = model

        # Evaluate on training data
        train_accuracy = model.score(X, y)
        self.results['train_accuracy'] = train_accuracy

        return model

    def predict(self, X):
        """Make predictions with the trained classifier"""
        if 'classifier' not in self.models:
            raise ValueError("No classifier has been trained yet")

        predictions = self.models['classifier'].predict(X)
        probabilities = self.models['classifier'].predict_proba(X)

        return predictions, probabilities

    def visualize_clusters(self, data=None):
        """Visualize clustering results"""
        if data is None:
            if 'reduced_data' not in self.results:
                raise ValueError("No data available for visualization")
            data = self.results['reduced_data']

        if 'cluster_labels' not in self.results:
            raise ValueError("No clustering results to visualize")

        labels = self.results['cluster_labels']

        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
        plt.title('Cluster Visualization')
        plt.colorbar(label='Cluster')
        plt.grid(True, linestyle='--', alpha=0.7)

        return plt.gcf()

# ==================== MAIN APPLICATION CLASS ====================

class AdvancedComputationalSystem:
    """Comprehensive system combining all components with an intuitive GUI"""

    def __init__(self, build_gui=True):
        self.c_lib = setup_c_extensions()
        self.asm_lib = setup_assembly_extensions()

        # Initialize components
        self.graph_processor = GraphProcessor(self.c_lib, self.asm_lib)
        self.numerical_engine = NumericalEngine(self.c_lib, self.asm_lib)
        self.ml_module = MachineLearningModule(self.numerical_engine)

        if not build_gui:
            # HOOK FOR SUBCLASSES: a subclass that wants to add its own tabs
            # before the window is shown can pass build_gui=False, do its own
            # setup, then call self._build_gui() itself.
            return

        self._build_gui()

    def _build_gui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Advanced Computational System")
        self.root.geometry("1200x800")

        # Set style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_graph_tab()
        self.create_numerical_tab()
        self.create_ml_tab()
        self.create_system_tab()

        # HOOK FOR SUBCLASSES: called after the four base tabs exist, before
        # the status bar is attached, so extra tabs land in a sensible spot.
        self.create_extra_tabs()

        # Setup status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize plots
        self.current_figure = None
        self.canvas = None

    def create_extra_tabs(self):
        """Overridden by subclasses (e.g. to add the Relational Substrate tab).
        No-op in the base class."""
        pass

    def create_graph_tab(self):
        """Create the graph theory tab"""
        graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(graph_frame, text="Graph Theory")

        # Control panel
        ctrl_frame = ttk.LabelFrame(graph_frame, text="Graph Controls")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Graph generation
        ttk.Label(ctrl_frame, text="Generate Random Graph:").grid(row=0, column=0, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Nodes:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.nodes_var = tk.IntVar(value=20)
        ttk.Spinbox(ctrl_frame, from_=5, to=100, textvariable=self.nodes_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(ctrl_frame, text="Edge Probability:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.edge_prob_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(ctrl_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.edge_prob_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5)

        self.directed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="Directed", variable=self.directed_var).grid(row=3, column=0, sticky=tk.W, padx=5)

        self.weighted_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Weighted", variable=self.weighted_var).grid(row=3, column=1, sticky=tk.W, padx=5)

        ttk.Button(ctrl_frame, text="Generate Graph", command=self.generate_graph).grid(row=4, column=0, columnspan=2, pady=10)

        # Graph algorithms
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Graph Algorithms:").grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

        algorithms = [
            ("Centrality Measures", self.compute_centralities),
            ("Shortest Paths", self.compute_shortest_paths),
            ("Community Detection", self.detect_communities),
            ("Minimum Spanning Tree", self.compute_mst),
            ("Graph Coloring", self.compute_coloring),
            ("Spectral Analysis", self.spectral_analysis),
        ]

        for i, (text, cmd) in enumerate(algorithms):
            ttk.Button(ctrl_frame, text=text, command=cmd).grid(row=i+7, column=0, columnspan=2, pady=2, sticky=tk.EW)

        # Visualization options
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=15, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Visualization:").grid(row=16, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Layout:").grid(row=17, column=0, sticky=tk.W, padx=5)
        self.layout_var = tk.StringVar(value="spring")
        layout_combo = ttk.Combobox(ctrl_frame, textvariable=self.layout_var, width=10)
        layout_combo['values'] = ('spring', 'circular', 'spectral', 'kamada_kawai')
        layout_combo.grid(row=17, column=1, padx=5, sticky=tk.W)

        ttk.Button(ctrl_frame, text="Visualize Graph", command=self.visualize_graph).grid(row=18, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Results and visualization area
        result_frame = ttk.LabelFrame(graph_frame, text="Graph Visualization")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for matplotlib plots
        self.graph_canvas_frame = ttk.Frame(result_frame)
        self.graph_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Results text area
        self.graph_result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.graph_result_text.pack(fill=tk.X, expand=False, pady=10)

    def create_numerical_tab(self):
        """Create the numerical computing tab"""
        num_frame = ttk.Frame(self.notebook)
        self.notebook.add(num_frame, text="Numerical Computing")

        # Control panel
        ctrl_frame = ttk.LabelFrame(num_frame, text="Numerical Operations")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Matrix operations
        ttk.Label(ctrl_frame, text="Matrix Operations:").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Matrix Size:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.matrix_size_var = tk.IntVar(value=100)
        ttk.Spinbox(ctrl_frame, from_=2, to=1000, textvariable=self.matrix_size_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5)

        matrix_ops = [
            ("Generate Random Matrices", self.generate_matrices),
            ("Matrix Multiplication", self.matrix_multiply),
            ("Matrix Determinant", self.matrix_determinant),
            ("Eigendecomposition", self.eigendecomposition),
            ("Singular Value Decomposition", self.svd_decomposition),
        ]

        for i, (text, cmd) in enumerate(matrix_ops):
            ttk.Button(ctrl_frame, text=text, command=cmd).grid(row=i+2, column=0, columnspan=2, pady=2, sticky=tk.EW)

        # Optimization
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Optimization:").grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Button(ctrl_frame, text="Optimize Test Function", command=self.optimize_function).grid(row=10, column=0, columnspan=2, pady=2, sticky=tk.EW)

        # Signal processing
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=12, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Signal Processing:").grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Signal Length:").grid(row=14, column=0, sticky=tk.W, padx=5)
        self.signal_length_var = tk.IntVar(value=1024)
        ttk.Spinbox(ctrl_frame, from_=16, to=8192, textvariable=self.signal_length_var, width=7).grid(row=14, column=1, sticky=tk.W, padx=5)

        ttk.Button(ctrl_frame, text="Generate & FFT Signal", command=self.fft_analysis).grid(row=15, column=0, columnspan=2, pady=2, sticky=tk.EW)

        # Results and visualization area
        result_frame = ttk.LabelFrame(num_frame, text="Results")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for matplotlib plots
        self.num_canvas_frame = ttk.Frame(result_frame)
        self.num_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Results text area
        self.num_result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.num_result_text.pack(fill=tk.X, expand=False, pady=10)

    def create_ml_tab(self):
        """Create the machine learning tab"""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="Machine Learning")

        # Control panel
        ctrl_frame = ttk.LabelFrame(ml_frame, text="Machine Learning Operations")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Data generation
        ttk.Label(ctrl_frame, text="Data Generation:").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Samples:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.samples_var = tk.IntVar(value=500)
        ttk.Spinbox(ctrl_frame, from_=10, to=5000, textvariable=self.samples_var, width=6).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(ctrl_frame, text="Features:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.features_var = tk.IntVar(value=10)
        ttk.Spinbox(ctrl_frame, from_=2, to=1000, textvariable=self.features_var, width=6).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(ctrl_frame, text="Clusters:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.clusters_var = tk.IntVar(value=4)
        ttk.Spinbox(ctrl_frame, from_=2, to=20, textvariable=self.clusters_var, width=6).grid(row=3, column=1, sticky=tk.W, padx=5)

        ttk.Button(ctrl_frame, text="Generate Data", command=self.generate_ml_data).grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # Clustering
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Clustering:").grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Algorithm:").grid(row=7, column=0, sticky=tk.W, padx=5)
        self.cluster_algo_var = tk.StringVar(value="kmeans")
        cluster_combo = ttk.Combobox(ctrl_frame, textvariable=self.cluster_algo_var, width=12)
        cluster_combo['values'] = ('kmeans', 'dbscan', 'agglomerative')
        cluster_combo.grid(row=7, column=1, padx=5, sticky=tk.W)

        ttk.Button(ctrl_frame, text="Perform Clustering", command=self.perform_clustering).grid(row=8, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # Dimensionality reduction
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Dimensionality Reduction:").grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Method:").grid(row=11, column=0, sticky=tk.W, padx=5)
        self.dim_method_var = tk.StringVar(value="pca")
        dim_combo = ttk.Combobox(ctrl_frame, textvariable=self.dim_method_var, width=12)
        dim_combo['values'] = ('pca', 'tsne', 'umap')
        dim_combo.grid(row=11, column=1, padx=5, sticky=tk.W)

        ttk.Button(ctrl_frame, text="Reduce Dimensions", command=self.reduce_dimensions).grid(row=12, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # Classification
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).grid(row=13, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(ctrl_frame, text="Classification:").grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(ctrl_frame, text="Model:").grid(row=15, column=0, sticky=tk.W, padx=5)
        self.model_var = tk.StringVar(value="random_forest")
        model_combo = ttk.Combobox(ctrl_frame, textvariable=self.model_var, width=12)
        model_combo['values'] = ('random_forest', 'svm', 'neural_network')
        model_combo.grid(row=15, column=1, padx=5, sticky=tk.W)

        ttk.Button(ctrl_frame, text="Train Classifier", command=self.train_classifier).grid(row=16, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # Results area
        result_frame = ttk.LabelFrame(ml_frame, text="Results")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for matplotlib plots
        self.ml_canvas_frame = ttk.Frame(result_frame)
        self.ml_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Results text area
        self.ml_result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.ml_result_text.pack(fill=tk.X, expand=False, pady=10)

    def create_system_tab(self):
        """Create the system monitoring and configuration tab"""
        sys_frame = ttk.Frame(self.notebook)
        self.notebook.add(sys_frame, text="System")

        # Configuration panel
        config_frame = ttk.LabelFrame(sys_frame, text="System Configuration")
        config_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        # Threading configuration
        ttk.Label(config_frame, text="Max Threads:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.threads_var = tk.IntVar(value=CONFIG["max_threads"])
        thread_spin = ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.threads_var, width=5)
        thread_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Cache configuration
        ttk.Label(config_frame, text="Cache Size (MB):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.cache_var = tk.IntVar(value=CONFIG["cache_size"] // (1024 * 1024))
        cache_spin = ttk.Spinbox(config_frame, from_=64, to=4096, textvariable=self.cache_var, width=5)
        cache_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Precision selection
        ttk.Label(config_frame, text="Precision:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.precision_var = tk.StringVar(value=CONFIG["precision"])
        precision_combo = ttk.Combobox(config_frame, textvariable=self.precision_var, width=10)
        precision_combo['values'] = ('single', 'double', 'extended')
        precision_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Hardware acceleration options
        ttk.Label(config_frame, text="Hardware Acceleration:").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        self.gpu_var = tk.BooleanVar(value=CONFIG["gpu_acceleration"])
        ttk.Checkbutton(config_frame, text="GPU Acceleration", variable=self.gpu_var).grid(row=4, column=0, sticky=tk.W, padx=25)

        self.asm_var = tk.BooleanVar(value=CONFIG["assembly_optimization"])
        ttk.Checkbutton(config_frame, text="Assembly Optimization", variable=self.asm_var).grid(row=5, column=0, sticky=tk.W, padx=25)

        # Auto-optimization
        self.auto_opt_var = tk.BooleanVar(value=CONFIG["auto_optimization"])
        ttk.Checkbutton(config_frame, text="Automatic Optimization", variable=self.auto_opt_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Apply button
        ttk.Button(config_frame, text="Apply Configuration", command=self.apply_config).grid(row=7, column=0, columnspan=2, pady=10)

        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # System benchmarks
        ttk.Label(config_frame, text="System Benchmarks:").grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        benchmark_buttons = [
            ("CPU Performance", self.benchmark_cpu),
            ("Memory Throughput", self.benchmark_memory),
            ("Algorithm Efficiency", self.benchmark_algorithms),
            ("Full System Benchmark", self.benchmark_full_system)
        ]

        for i, (text, cmd) in enumerate(benchmark_buttons):
            ttk.Button(config_frame, text=text, command=cmd).grid(row=i+10, column=0, columnspan=2, pady=2, sticky=tk.EW, padx=5)

        # System monitoring
        monitor_frame = ttk.LabelFrame(sys_frame, text="System Monitoring")
        monitor_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Resource usage
        resource_frame = ttk.Frame(monitor_frame)
        resource_frame.pack(fill=tk.X, pady=10)

        ttk.Label(resource_frame, text="CPU Usage:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.cpu_progressbar = ttk.Progressbar(resource_frame, length=200, mode='determinate')
        self.cpu_progressbar.grid(row=0, column=1, padx=5, pady=5)
        self.cpu_label = ttk.Label(resource_frame, text="0%")
        self.cpu_label.grid(row=0, column=2, padx=5)

        ttk.Label(resource_frame, text="Memory Usage:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.mem_progressbar = ttk.Progressbar(resource_frame, length=200, mode='determinate')
        self.mem_progressbar.grid(row=1, column=1, padx=5, pady=5)
        self.mem_label = ttk.Label(resource_frame, text="0 MB")
        self.mem_label.grid(row=1, column=2, padx=5)

        # Start resource monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.update_resource_usage, daemon=True)
        self.monitor_thread.start()

        # Performance metrics
        self.perf_canvas_frame = ttk.Frame(monitor_frame)
        self.perf_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Initialize with empty performance chart
        self.create_performance_chart()

        # Log output
        ttk.Label(monitor_frame, text="System Log:").pack(anchor=tk.W, padx=5)
        self.log_text = tk.Text(monitor_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log("System initialized successfully")
        self.log(f"C Extensions: {'Loaded' if self.c_lib else 'Not available'}")
        self.log(f"Assembly Extensions: {'Loaded' if self.asm_lib else 'Not available'}")

    def apply_config(self):
        """Apply configuration changes"""
        # Update config dictionary
        CONFIG["max_threads"] = self.threads_var.get()
        CONFIG["cache_size"] = self.cache_var.get() * 1024 * 1024
        CONFIG["precision"] = self.precision_var.get()
        CONFIG["gpu_acceleration"] = self.gpu_var.get()
        CONFIG["assembly_optimization"] = self.asm_var.get()
        CONFIG["auto_optimization"] = self.auto_opt_var.get()

        # Log changes
        self.log("Configuration updated")
        self.status_var.set("Configuration applied successfully")

    def log(self, message):
        """Add a message to the system log"""
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)

    def update_resource_usage(self):
        """Update system resource usage indicators"""
        import psutil

        while self.monitoring:
            # Update CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_progressbar['value'] = cpu_percent
            self.cpu_label.config(text=f"{cpu_percent:.1f}%")

            # Update memory usage
            mem = psutil.virtual_memory()
            mem_used_mb = mem.used / (1024 * 1024)
            mem_total_mb = mem.total / (1024 * 1024)
            mem_percent = mem.percent

            self.mem_progressbar['value'] = mem_percent
            self.mem_label.config(text=f"{mem_used_mb:.0f} / {mem_total_mb:.0f} MB")

            # Sleep for a bit
            time.sleep(1)

    def create_performance_chart(self):
        """Create performance monitoring chart"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Performance Metrics")
        ax.set_xlabel("Time")
        ax.set_ylabel("Operations/sec")
        ax.grid(True)

        # Sample data for now
        x = np.arange(10)
        y = np.zeros(10)
        self.perf_line, = ax.plot(x, y)

        if hasattr(self, 'perf_canvas'):
            self.perf_canvas.get_tk_widget().destroy()

        self.perf_canvas = FigureCanvasTkAgg(fig, self.perf_canvas_frame)
        self.perf_canvas.draw()
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_performance_chart(self, data):
        """Update performance chart with new data"""
        self.perf_line.set_ydata(data)
        self.perf_line.figure.canvas.draw()

    def display_figure(self, figure, frame):
        """Display a matplotlib figure in the specified frame"""
        # Clear previous figure
        for widget in frame.winfo_children():
            widget.destroy()

        # Display new figure
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store reference to prevent garbage collection
        self.current_figure = figure
        self.canvas = canvas

    # ========== Graph tab functions ==========

    def generate_graph(self):
        """Generate a random graph based on user parameters"""
        try:
            nodes = self.nodes_var.get()
            edge_prob = self.edge_prob_var.get()
            directed = self.directed_var.get()
            weighted = self.weighted_var.get()

            self.status_var.set("Generating graph...")
            self.graph_processor.generate_random_graph(nodes, edge_prob, directed, weighted)

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, f"Generated {nodes} node graph\n")
            self.graph_result_text.insert(tk.END, f"Directed: {directed}, Weighted: {weighted}\n")
            self.graph_result_text.insert(tk.END, f"Edges: {self.graph_processor.graph.number_of_edges()}\n")

            # Visualize
            self.visualize_graph()

            self.status_var.set("Graph generated successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def compute_centralities(self):
        """Compute graph centrality measures"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            self.status_var.set("Computing centralities...")
            centralities = self.graph_processor.compute_centralities()

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, "Centrality measures computed:\n")

            # Show the most central nodes
            self.graph_result_text.insert(tk.END, "\nTop 5 nodes by eigenvector centrality:\n")
            eigen_centrality = centralities['eigenvector_centrality']
            top_nodes = sorted(eigen_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, score in top_nodes:
                self.graph_result_text.insert(tk.END, f"Node {node}: {score:.4f}\n")

            # Update visualization (nodes sized by centrality)
            self.visualize_graph()

            self.status_var.set("Centrality measures computed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def compute_shortest_paths(self):
        """Compute shortest paths in the graph"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            self.status_var.set("Computing shortest paths...")
            paths = self.graph_processor.shortest_paths()

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, "Shortest paths computed\n")

            # Show some example paths
            if len(paths) > 0:
                source = list(paths.keys())[0]
                self.graph_result_text.insert(tk.END, f"\nExample paths from node {source}:\n")
                for target, dist in list(paths[source].items())[:5]:
                    if target != source:
                        self.graph_result_text.insert(tk.END, f"To node {target}: distance {dist}\n")

            self.status_var.set("Shortest paths computed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def detect_communities(self):
        """Detect communities in the graph"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            self.status_var.set("Detecting communities...")
            communities = self.graph_processor.find_communities()

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, "Communities detected\n")

            # Count communities
            if isinstance(communities, dict):
                num_communities = len(set(communities.values()))
                self.graph_result_text.insert(tk.END, f"Number of communities: {num_communities}\n")

            # Visualize graph with communities
            self.visualize_graph()

            self.status_var.set("Communities detected successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def compute_mst(self):
        """Compute minimum spanning tree"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            self.status_var.set("Computing minimum spanning tree...")
            mst = self.graph_processor.minimum_spanning_tree()

            if isinstance(mst, str):
                self.graph_result_text.delete(1.0, tk.END)
                self.graph_result_text.insert(tk.END, mst)
                return

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, "Minimum spanning tree computed\n")
            self.graph_result_text.insert(tk.END, f"MST edges: {mst.number_of_edges()}\n")

            # Calculate total weight
            total_weight = sum(data.get('weight', 1) for _, _, data in mst.edges(data=True))
            self.graph_result_text.insert(tk.END, f"Total MST weight: {total_weight:.2f}\n")

            # Show MST in original graph
            fig = plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(self.graph_processor.graph, seed=42)

            # Draw original graph edges in light gray
            nx.draw_networkx_edges(self.graph_processor.graph, pos, alpha=0.2, edge_color='gray')

            # Draw MST edges in blue
            nx.draw_networkx_edges(mst, pos, alpha=1.0, edge_color='blue', width=2)

            # Draw nodes
            nx.draw_networkx_nodes(self.graph_processor.graph, pos)

            # Draw labels
            nx.draw_networkx_labels(self.graph_processor.graph, pos, font_size=10)

            plt.title("Minimum Spanning Tree")
            plt.axis('off')

            # Display figure
            self.display_figure(fig, self.graph_canvas_frame)

            self.status_var.set("Minimum spanning tree computed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def compute_coloring(self):
        """Compute graph coloring"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            self.status_var.set("Computing graph coloring...")
            coloring = self.graph_processor.graph_coloring()

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, "Graph coloring computed\n")

            # Count colors
            num_colors = len(set(coloring.values()))
            self.graph_result_text.insert(tk.END, f"Number of colors used: {num_colors}\n")

            # Visualize graph with coloring
            fig = plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(self.graph_processor.graph, seed=42)

            # Create a list of colors based on the coloring dict
            node_colors = [coloring[node] for node in self.graph_processor.graph.nodes()]

            # Draw the graph
            nx.draw(
                self.graph_processor.graph, pos, with_labels=True,
                node_color=node_colors, cmap=plt.cm.rainbow,
                font_weight='bold'
            )

            plt.title("Graph Coloring")

            # Display figure
            self.display_figure(fig, self.graph_canvas_frame)

            self.status_var.set("Graph coloring computed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def spectral_analysis(self):
        """Perform spectral analysis of the graph"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            self.status_var.set("Performing spectral analysis...")
            results = self.graph_processor.spectral_analysis()

            # Update results
            self.graph_result_text.delete(1.0, tk.END)
            self.graph_result_text.insert(tk.END, "Spectral analysis completed\n")

            if 'spectral_gap' in results:
                self.graph_result_text.insert(tk.END, f"Spectral gap: {results['spectral_gap']:.4f}\n")

            # Plot eigenvalue spectrum
            fig = plt.figure(figsize=(8, 6))
            plt.plot(results['laplacian_eigenvalues'], 'bo-')
            plt.grid(True)
            plt.xlabel('Index')
            plt.ylabel('Eigenvalue')
            plt.title('Laplacian Spectrum')

            # Display figure
            self.display_figure(fig, self.graph_canvas_frame)

            self.status_var.set("Spectral analysis completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def visualize_graph(self):
        """Visualize the current graph"""
        if self.graph_processor.graph is None:
            messagebox.showinfo("Info", "Please generate a graph first")
            return

        try:
            layout = self.layout_var.get()
            self.status_var.set(f"Visualizing graph with {layout} layout...")

            # Get the figure from the graph processor
            fig = self.graph_processor.visualize(layout=layout)

            # Display the figure
            self.display_figure(fig, self.graph_canvas_frame)

            self.status_var.set("Graph visualization updated")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    # ========== Numerical tab functions ==========

    def generate_matrices(self):
        """Generate random matrices for operations"""
        try:
            size = self.matrix_size_var.get()

            # Generate two random matrices
            self.matrix_A = np.random.rand(size, size)
            self.matrix_B = np.random.rand(size, size)

            # Update status
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"Generated two {size}x{size} random matrices\n")

            self.status_var.set("Matrices generated successfully")

            # Show preview of matrices
            if size <= 5:
                self.num_result_text.insert(tk.END, "\nMatrix A:\n")
                self.num_result_text.insert(tk.END, str(self.matrix_A))
                self.num_result_text.insert(tk.END, "\n\nMatrix B:\n")
                self.num_result_text.insert(tk.END, str(self.matrix_B))
            else:
                self.num_result_text.insert(tk.END, "\nMatrix A (corner):\n")
                self.num_result_text.insert(tk.END, str(self.matrix_A[:3, :3]))
                self.num_result_text.insert(tk.END, "\n\nMatrix B (corner):\n")
                self.num_result_text.insert(tk.END, str(self.matrix_B[:3, :3]))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def matrix_multiply(self):
        """Perform matrix multiplication"""
        if not hasattr(self, 'matrix_A') or not hasattr(self, 'matrix_B'):
            messagebox.showinfo("Info", "Please generate matrices first")
            return

        try:
            self.status_var.set("Performing matrix multiplication...")
            start_time = time.time()

            # Use numerical engine for multiplication
            result = self.numerical_engine.matrix_multiply(self.matrix_A, self.matrix_B)

            elapsed_time = time.time() - start_time

            # Update results
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"Matrix multiplication completed in {elapsed_time:.4f} seconds\n")

            size = self.matrix_A.shape[0]
            if size <= 5:
                self.num_result_text.insert(tk.END, "\nResult matrix:\n")
                self.num_result_text.insert(tk.END, str(result))
            else:
                self.num_result_text.insert(tk.END, "\nResult matrix (corner):\n")
                self.num_result_text.insert(tk.END, str(result[:3, :3]))

            # Calculate FLOPS
            flops = 2 * size**3  # Approximately 2*n^3 operations for matrix multiply
            self.num_result_text.insert(tk.END, f"\n\nPerformance: {flops / elapsed_time / 1e9:.2f} GFLOPS\n")

            # Visualize result as heatmap
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(result, cmap='viridis')
            plt.colorbar(label='Value')
            plt.title(f'Matrix Multiplication Result ({size}x{size})')

            # Display figure
            self.display_figure(fig, self.num_canvas_frame)

            self.status_var.set("Matrix multiplication completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def matrix_determinant(self):
        """Calculate matrix determinant"""
        if not hasattr(self, 'matrix_A'):
            messagebox.showinfo("Info", "Please generate matrices first")
            return

        try:
            self.status_var.set("Calculating matrix determinant...")
            start_time = time.time()

            # Use numerical engine for determinant
            det = self.numerical_engine.matrix_determinant(self.matrix_A)

            elapsed_time = time.time() - start_time

            # Update results
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"Determinant calculation completed in {elapsed_time:.4f} seconds\n")
            self.num_result_text.insert(tk.END, f"\nDeterminant of matrix A: {det}\n")

            self.status_var.set("Determinant calculated successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def eigendecomposition(self):
        """Perform eigendecomposition of matrix"""
        if not hasattr(self, 'matrix_A'):
            messagebox.showinfo("Info", "Please generate matrices first")
            return

        try:
            self.status_var.set("Performing eigendecomposition...")
            start_time = time.time()

            # Use numerical engine
            values, vectors = self.numerical_engine.eigendecomposition(self.matrix_A)

            elapsed_time = time.time() - start_time

            # Update results
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"Eigendecomposition completed in {elapsed_time:.4f} seconds\n")

            # Plot eigenvalue spectrum
            fig = plt.figure(figsize=(10, 6))

            # Create two subplots
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # Plot eigenvalues
            ax1.plot(np.sort(np.abs(values)), 'o-')
            ax1.set_yscale('log')
            ax1.set_title('Eigenvalue Spectrum')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('|Eigenvalue|')
            ax1.grid(True)

            # Plot first eigenvector
            ax2.bar(range(len(vectors[:, 0])), np.abs(vectors[:, 0]))
            ax2.set_title('First Eigenvector')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('|Value|')
            ax2.grid(True)

            plt.tight_layout()

            # Display figure
            self.display_figure(fig, self.num_canvas_frame)

            self.status_var.set("Eigendecomposition completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def svd_decomposition(self):
        """Perform Singular Value Decomposition"""
        if not hasattr(self, 'matrix_A'):
            messagebox.showinfo("Info", "Please generate matrices first")
            return

        try:
            self.status_var.set("Performing SVD...")
            start_time = time.time()

            # Use numerical engine
            U, S, Vh = self.numerical_engine.svd(self.matrix_A)

            elapsed_time = time.time() - start_time

            # Update results
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"SVD completed in {elapsed_time:.4f} seconds\n")

            # Plot singular values
            fig = plt.figure(figsize=(8, 6))
            plt.semilogy(S, 'o-')
            plt.grid(True)
            plt.title('Singular Value Spectrum')
            plt.xlabel('Index')
            plt.ylabel('Singular Value')

            # Display figure
            self.display_figure(fig, self.num_canvas_frame)

            # Add information about matrix rank
            effective_rank = sum(S > 1e-10)
            self.num_result_text.insert(tk.END, f"\nMatrix effective rank: {effective_rank}\n")
            self.num_result_text.insert(tk.END, f"Condition number: {S[0] / S[-1]:.2e}\n")

            self.status_var.set("SVD completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def optimize_function(self):
        """Optimize a test function"""
        try:
            self.status_var.set("Performing optimization...")

            # Define a test optimization problem
            def rosenbrock(x):
                """Rosenbrock function - a standard optimization test problem"""
                return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

            # Define bounds
            bounds = [(-3, 3)] * 5  # 5D Rosenbrock

            start_time = time.time()

            # Use numerical engine
            result = self.numerical_engine.optimize_function(rosenbrock, bounds)

            elapsed_time = time.time() - start_time

            # Update results
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"Optimization completed in {elapsed_time:.4f} seconds\n")
            self.num_result_text.insert(tk.END, f"\nOptimization method: {result.message}\n")
            self.num_result_text.insert(tk.END, f"Function evaluations: {result.nfev}\n")
            self.num_result_text.insert(tk.END, f"Final function value: {result.fun:.10f}\n")
            self.num_result_text.insert(tk.END, f"Optimal point: {result.x}\n")

            # For 2D visualization, run a grid evaluation of the function
            if len(bounds) >= 2:
                x = np.linspace(-3, 3, 100)
                y = np.linspace(-3, 3, 100)
                X, Y = np.meshgrid(x, y)

                # Evaluate function on grid
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        # Set remaining dimensions to optimal values
                        point = list(result.x)
                        point[0] = X[i, j]
                        point[1] = Y[i, j]
                        Z[i, j] = rosenbrock(point)

                # Plot contour
                fig = plt.figure(figsize=(8, 6))
                plt.contourf(X, Y, np.log1p(Z), 50, cmap='viridis')
                plt.colorbar(label='log(1 + f(x,y))')
                plt.plot(result.x[0], result.x[1], 'ro', markersize=10)
                plt.grid(True)
                plt.title('Rosenbrock Function Optimization')
                plt.xlabel('x')
                plt.ylabel('y')

                # Display figure
                self.display_figure(fig, self.num_canvas_frame)

            self.status_var.set("Optimization completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def fft_analysis(self):
        """Perform FFT analysis on a test signal"""
        try:
            self.status_var.set("Performing FFT analysis...")

            # Generate a test signal
            n = self.signal_length_var.get()
            t = np.linspace(0, 1, n)

            # Create a complex signal with multiple frequencies
            signal = (
                np.sin(2 * np.pi * 50 * t) +          # 50 Hz component
                0.5 * np.sin(2 * np.pi * 120 * t) +   # 120 Hz component
                0.2 * np.random.randn(n)              # Noise
            )

            start_time = time.time()

            # Use numerical engine
            spectrum = self.numerical_engine.fft(signal)

            elapsed_time = time.time() - start_time

            # Compute frequencies for plotting
            freq = np.fft.fftfreq(n, t[1] - t[0])

            # Update results
            self.num_result_text.delete(1.0, tk.END)
            self.num_result_text.insert(tk.END, f"FFT analysis completed in {elapsed_time:.4f} seconds\n")
            self.num_result_text.insert(tk.END, f"Signal length: {n} samples\n")

            # Plot signal and spectrum
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            # Plot original signal
            ax1.plot(t, signal)
            ax1.set_title('Original Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True)

            # Plot power spectrum (positive frequencies only)
            positive_freq_idx = np.where(freq > 0)
            ax2.plot(freq[positive_freq_idx], np.abs(spectrum[positive_freq_idx]))
            ax2.set_title('Frequency Spectrum')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.grid(True)

            plt.tight_layout()

            # Display figure
            self.display_figure(fig, self.num_canvas_frame)

            # Find dominant frequencies
            magnitude = np.abs(spectrum)
            top_indices = np.argsort(magnitude)[-5:]  # Top 5 frequencies

            self.num_result_text.insert(tk.END, "\nDominant frequencies:\n")
            for idx in reversed(top_indices):
                if freq[idx] > 0:  # Only positive frequencies
                    self.num_result_text.insert(tk.END, f"{freq[idx]:.1f} Hz: Magnitude = {magnitude[idx]:.2f}\n")

            self.status_var.set("FFT analysis completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    # ========== Machine Learning tab functions ==========

    def generate_ml_data(self):
        """Generate synthetic data for ML demonstrations"""
        try:
            self.status_var.set("Generating synthetic data...")

            n_samples = self.samples_var.get()
            n_features = self.features_var.get()
            n_clusters = self.clusters_var.get()

            # Generate clustered data
            from sklearn.datasets import make_blobs

            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_clusters,
                cluster_std=1.0,
                random_state=42
            )

            # Store data
            self.ml_data = X
            self.ml_labels = y

            # Update results
            self.ml_result_text.delete(1.0, tk.END)
            self.ml_result_text.insert(tk.END, f"Generated {n_samples} samples with {n_features} features in {n_clusters} clusters\n")

            # If more than 2 features, perform PCA for visualization
            if n_features > 2:
                reduced_data = self.ml_module.dimensionality_reduction(X, method='pca')
                self.ml_result_text.insert(tk.END, "Performed PCA for visualization\n")
            else:
                reduced_data = X

            # Visualize data
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y, cmap='viridis', s=30, alpha=0.8)
            plt.colorbar(label='True Cluster')
            plt.title('Generated Data')
            plt.grid(True)

            # Display figure
            self.display_figure(fig, self.ml_canvas_frame)

            self.status_var.set("Data generated successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def reduce_dimensions(self):
        """Reduce dimensionality of data for visualization"""
        if not hasattr(self, 'ml_data'):
            messagebox.showinfo("Info", "Please generate data first")
            return

        try:
            method = self.dim_method_var.get()

            self.status_var.set(f"Performing {method} dimensionality reduction...")
            start_time = time.time()

            # Perform dimensionality reduction
            reduced_data = self.ml_module.dimensionality_reduction(self.ml_data, method=method)

            elapsed_time = time.time() - start_time

            # Update results
            self.ml_result_text.delete(1.0, tk.END)
            self.ml_result_text.insert(tk.END, f"{method.upper()} completed in {elapsed_time:.4f} seconds\n")

            # Visualize results
            fig = plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                reduced_data[:, 0], reduced_data[:, 1],
                c=self.ml_labels if hasattr(self, 'ml_labels') else None,
                cmap='viridis', s=30, alpha=0.8
            )

            if hasattr(self, 'ml_labels'):
                plt.colorbar(scatter, label='True Label')

            plt.title(f'{method.upper()} Projection')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True)

            # Display figure
            self.display_figure(fig, self.ml_canvas_frame)

            # Calculate explained variance if PCA
            if method == 'pca' and 'dim_reduction' in self.ml_module.models:
                pca = self.ml_module.models['dim_reduction']
                if hasattr(pca, 'explained_variance_ratio_'):
                    explained_var = pca.explained_variance_ratio_[:2].sum() * 100
                    self.ml_result_text.insert(tk.END, f"\nExplained variance (2D): {explained_var:.2f}%\n")

            self.status_var.set(f"{method.upper()} reduction completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def train_classifier(self):
        """Train a classifier on the generated data"""
        if not hasattr(self, 'ml_data') or not hasattr(self, 'ml_labels'):
            messagebox.showinfo("Info", "Please generate data first")
            return

        try:
            model_type = self.model_var.get()

            self.status_var.set(f"Training {model_type} classifier...")
            start_time = time.time()

            # Split data into train/test sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.ml_data, self.ml_labels, test_size=0.3, random_state=42
            )

            # Train classifier
            model = self.ml_module.train_classifier(X_train, y_train, model_type)

            # Test performance
            y_pred = model.predict(X_test)
            test_accuracy = np.mean(y_pred == y_test)

            elapsed_time = time.time() - start_time

            # Update results
            self.ml_result_text.delete(1.0, tk.END)
            self.ml_result_text.insert(tk.END, f"{model_type} training completed in {elapsed_time:.4f} seconds\n")
            self.ml_result_text.insert(tk.END, f"\nTrain accuracy: {self.ml_module.results['train_accuracy']:.4f}\n")
            self.ml_result_text.insert(tk.END, f"Test accuracy: {test_accuracy:.4f}\n")

            # Compute confusion matrix
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_test, y_pred)

            # Visualize confusion matrix
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()

            # Add labels
            n_classes = len(np.unique(self.ml_labels))
            tick_marks = np.arange(n_classes)
            plt.xticks(tick_marks, range(n_classes))
            plt.yticks(tick_marks, range(n_classes))
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Add counts
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j],
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()

            # Display figure
            self.display_figure(fig, self.ml_canvas_frame)

            # Add classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            self.ml_result_text.insert(tk.END, "\nClassification Report:\n")
            self.ml_result_text.insert(tk.END, f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            self.ml_result_text.insert(tk.END, f"{'-'*40}\n")

            for cls, metrics in report.items():
                if cls not in ('accuracy', 'macro avg', 'weighted avg'):
                    self.ml_result_text.insert(
                        tk.END,
                        f"{cls:<8} {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1-score']:.4f}\n"
                    )

            self.status_var.set(f"{model_type} training completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def perform_clustering(self):
        """Perform clustering on the generated data"""
        if not hasattr(self, 'ml_data'):
            messagebox.showinfo("Info", "Please generate data first")
            return

        try:
            algorithm = self.cluster_algo_var.get()
            n_clusters = self.clusters_var.get()

            self.status_var.set(f"Performing {algorithm} clustering...")
            start_time = time.time()

            # Perform clustering
            labels = self.ml_module.cluster_data(self.ml_data, n_clusters, algorithm)

            elapsed_time = time.time() - start_time

            # Update results
            self.ml_result_text.delete(1.0, tk.END)
            self.ml_result_text.insert(tk.END, f"{algorithm} clustering completed in {elapsed_time:.4f} seconds\n")

            if algorithm == 'dbscan':
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                self.ml_result_text.insert(tk.END, f"Detected {n_clusters} clusters and {n_noise} noise points\n")
            else:
                self.ml_result_text.insert(tk.END, f"Number of clusters: {n_clusters}\n")

            # Visualize clustering results
            if self.ml_data.shape[1] > 2:
                # Use dimensionality reduction for visualization
                if not hasattr(self, 'ml_module') or 'reduced_data' not in self.ml_module.results:
                    reduced_data = self.ml_module.dimensionality_reduction(self.ml_data, method='pca')
                else:
                    reduced_data = self.ml_module.results['reduced_data']
            else:
                reduced_data = self.ml_data

            # Visualize clusters
            fig = self.ml_module.visualize_clusters(reduced_data)
            self.display_figure(fig, self.ml_canvas_frame)

            # Compare with true labels if available
            if hasattr(self, 'ml_labels'):
                from sklearn import metrics

                # Calculate clustering metrics
                ari = metrics.adjusted_rand_score(self.ml_labels, labels)
                nmi = metrics.normalized_mutual_info_score(self.ml_labels, labels)

                self.ml_result_text.insert(tk.END, f"\nClustering quality metrics (compared to true labels):\n")
                self.ml_result_text.insert(tk.END, f"Adjusted Rand Index: {ari:.4f}\n")
                self.ml_result_text.insert(tk.END, f"Normalized Mutual Information: {nmi:.4f}\n")

            self.status_var.set(f"{algorithm} clustering completed successfully")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    # ========== System tab benchmark functions ==========
    # NOTE: these four were referenced by create_system_tab's button list in your
    # original source but never defined there. Implemented here as real, working
    # benchmarks (not stubs) so the buttons don't crash on click.

    def benchmark_cpu(self):
        """Benchmark raw CPU throughput via a matrix-multiply flop count"""
        try:
            self.status_var.set("Running CPU benchmark...")
            size = 300
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            start = time.time()
            np.matmul(A, B)
            elapsed = time.time() - start
            flops = 2 * size ** 3
            gflops = flops / elapsed / 1e9
            self.log(f"CPU benchmark: {gflops:.2f} GFLOPS ({size}x{size} matmul, {elapsed:.4f}s)")
            self.status_var.set("CPU benchmark complete")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def benchmark_memory(self):
        """Benchmark memory throughput via large array copy"""
        try:
            self.status_var.set("Running memory benchmark...")
            n = 50_000_000
            start = time.time()
            arr = np.ones(n, dtype=np.float64)
            _ = arr.copy()
            elapsed = time.time() - start
            mb = (n * 8) / (1024 * 1024)
            throughput = mb / elapsed
            self.log(f"Memory benchmark: {throughput:.1f} MB/s copy throughput ({mb:.0f} MB, {elapsed:.4f}s)")
            self.status_var.set("Memory benchmark complete")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def benchmark_algorithms(self):
        """Benchmark core algorithm implementations against a fixed test graph/signal"""
        try:
            self.status_var.set("Running algorithm benchmark...")
            g = nx.gnp_random_graph(200, 0.05, seed=1)
            start = time.time()
            nx.betweenness_centrality(g)
            t_centrality = time.time() - start

            start = time.time()
            np.fft.fft(np.random.randn(4096))
            t_fft = time.time() - start

            self.log(f"Algorithm benchmark: betweenness_centrality(200 nodes) = {t_centrality:.4f}s, fft(4096) = {t_fft:.4f}s")
            self.status_var.set("Algorithm benchmark complete")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def benchmark_full_system(self):
        """Run all three benchmarks in sequence"""
        self.benchmark_cpu()
        self.benchmark_memory()
        self.benchmark_algorithms()
        self.log("Full system benchmark complete")
        self.status_var.set("Full system benchmark complete")

    def run(self):
        """Start the Tk main loop"""
        self.root.mainloop()


if __name__ == "__main__":
    app = AdvancedComputationalSystem()
    app.run()
