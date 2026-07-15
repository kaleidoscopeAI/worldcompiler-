"""Headless smoke test: build the real Tk GUI under Xvfb, exercise the main
paths in each tab (including the new Relational Substrate tab), and report
any exceptions. Not a unit test suite -- a "does it actually run" check."""
import sys
import traceback

# Prevent any messagebox.* call from blocking the test in a modal wait_window
# loop (there's no human here to click OK) -- redirect to stderr instead.
from tkinter import messagebox
def _fake_box(title, message, *a, **kw):
    print(f"[messagebox suppressed] {title}: {message}", file=sys.stderr)
messagebox.showinfo = _fake_box
messagebox.showerror = _fake_box
messagebox.showwarning = _fake_box

from advanced_unified_system import UnifiedSystem

errors = []


def try_step(name, fn):
    try:
        fn()
        print(f"OK   - {name}")
    except Exception:
        print(f"FAIL - {name}")
        traceback.print_exc()
        errors.append(name)


app = UnifiedSystem()
print("OK   - UnifiedSystem() constructed (window + all 5 tabs built)")

# Graph tab
try_step("generate_graph", app.generate_graph)
try_step("compute_centralities", app.compute_centralities)
try_step("compute_shortest_paths", app.compute_shortest_paths)
try_step("detect_communities", app.detect_communities)
try_step("compute_mst", app.compute_mst)
try_step("compute_coloring", app.compute_coloring)
try_step("spectral_analysis", app.spectral_analysis)

# Numerical tab (small matrix to keep it fast)
app.matrix_size_var.set(20)
try_step("generate_matrices", app.generate_matrices)
try_step("matrix_multiply", app.matrix_multiply)
try_step("matrix_determinant", app.matrix_determinant)
try_step("eigendecomposition", app.eigendecomposition)
try_step("svd_decomposition", app.svd_decomposition)
try_step("optimize_function", app.optimize_function)
app.signal_length_var.set(256)
try_step("fft_analysis", app.fft_analysis)

# ML tab (small sample size to keep it fast)
app.samples_var.set(60)
app.features_var.set(4)
app.clusters_var.set(3)
try_step("generate_ml_data", app.generate_ml_data)
try_step("perform_clustering", app.perform_clustering)
try_step("reduce_dimensions", app.reduce_dimensions)
try_step("train_classifier", app.train_classifier)

# System tab benchmarks
try_step("benchmark_cpu", app.benchmark_cpu)
try_step("benchmark_memory", app.benchmark_memory)
try_step("benchmark_algorithms", app.benchmark_algorithms)

# Relational Substrate tab: run briefly, force a split, import to graph
import time
try_step("start_rel_sim", app.start_rel_sim)
time.sleep(2.0)
try_step("force_split", app.force_split)
time.sleep(1.0)
try_step("import_registry_to_graph", app.import_registry_to_graph)
try_step("stop_rel_sim", app.stop_rel_sim)

app.monitoring = False  # stop the resource-monitor thread
app.root.update()
app.root.destroy()

if errors:
    print(f"\n{len(errors)} STEP(S) FAILED: {errors}")
    sys.exit(1)
else:
    print("\nALL STEPS PASSED")
