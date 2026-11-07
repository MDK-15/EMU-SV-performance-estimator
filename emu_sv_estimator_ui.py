"""
Web UI for the EMU-SV performance estimator.

Needs emu_sv_estimator.py in the same folder.
Run:  streamlit run emu_sv_estimator_ui.py
"""
import matplotlib.pyplot as plt
import streamlit as st

from emu_sv_estimator import (
    EFFICIENCIES,
    K_TABLE,
    estimate_k,
    fmt_time,
    max_qubits,
    memory_bytes,
    runtime_seconds,
)

st.set_page_config(page_title="EMU-SV performance estimator", layout="wide")
st.title("EMU-SV performance estimator")
st.caption("Noiseless state-vector mode only. Estimates, not measurements.")

with st.sidebar:
    st.header("Device")
    memory_gb = st.number_input("Memory capacity (GB)", min_value=0.1, value=40.0, step=1.0,
                                help="GPU memory for CUDA runs, or RAM for CPU runs.")
    bandwidth_gbs = st.number_input("Total memory bandwidth (GB/s)", min_value=1.0, value=1555.0, step=10.0,
                                    help="Whole-socket bandwidth of the CPU or GPU.")
    st.header("Emulation")
    duration_ns = st.number_input("Sequence duration T (ns)", min_value=1.0, value=3000.0, step=100.0)
    dt_ns = st.number_input("Time step dt (ns)", min_value=1.0, value=10.0, step=1.0,
                            help="Rounded up to 5, 10, 20 or 50 ns when estimating k.")
    n_traj = st.number_input("Number of trajectories", min_value=1, value=1, step=1,
                             help="1 for noiseless runs.")
    check_n = st.number_input("Check a specific qubit count (0 = skip)", min_value=0, value=0, step=1)

if dt_ns > max(K_TABLE):
    st.warning(f"dt = {dt_ns:g} ns is above the measured range ({max(K_TABLE)} ns). k is probably underestimated.")

n_max = max_qubits(memory_gb * 1e9, dt_ns)
if n_max == 0:
    st.error("Not even 1 qubit fits in this memory.")
    st.stop()

bw = bandwidth_gbs * 1e9


def runtimes(n):
    return {name: runtime_seconds(n, dt_ns, duration_ns, bw, eff, n_traj) for name, eff in EFFICIENCIES.items()}


# Headline numbers
c1, c2, c3 = st.columns(3)
c1.metric("Maximum qubits", n_max)
c2.metric("Krylov vectors k at that size", estimate_k(n_max, dt_ns))
c3.metric("Peak memory at that size", f"{memory_bytes(n_max, dt_ns) / 1e9:.1f} GB")


def runtime_table(n):
    rt = runtimes(n)
    return {"Case": [f"{name.capitalize()} ({EFFICIENCIES[name]:.0%} efficiency)" for name in rt],
            "Runtime": [fmt_time(t) for t in rt.values()]}


st.subheader(f"Runtime at {n_max} qubits")
st.table(runtime_table(n_max))

if check_n:
    st.subheader(f"Check: {check_n} qubits")
    if check_n > n_max:
        st.error(f"{check_n} qubits does not fit. It needs {memory_bytes(check_n, dt_ns) / 1e9:.1f} GB "
                 f"and the device has {memory_gb:g} GB.")
    else:
        st.write(f"k = {estimate_k(check_n, dt_ns)}, peak memory {memory_bytes(check_n, dt_ns) / 1e9:.2f} GB.")
        st.table(runtime_table(check_n))

# Graphs
ns = list(range(1, n_max + 1))
fig, (ax_mem, ax_rt) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_mem.semilogy(ns, [memory_bytes(n, dt_ns) / 1e9 for n in ns], marker="o")
ax_mem.axhline(memory_gb, color="gray", linestyle="--", label=f"Capacity ({memory_gb:g} GB)")
ax_mem.set(xlabel="Number of qubits N", ylabel="Peak memory (GB)", title="Memory consumption")
for name in EFFICIENCIES:
    ax_rt.semilogy(ns, [runtimes(n)[name] for n in ns], marker="o", label=name.capitalize())
ax_rt.set(xlabel="Number of qubits N", ylabel="Runtime (s)", title="Estimated runtime")
for ax in (ax_mem, ax_rt):
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
st.pyplot(fig)

st.caption("Runtime is a bandwidth model and is only meaningful at large N. At small N, GPU launch overhead "
           "and CPU caches dominate, so the graph underestimates there.")
