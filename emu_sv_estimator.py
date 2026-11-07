"""
EMU-SV performance estimator (noiseless state-vector mode only).

Given a device's memory capacity and bandwidth, plus the Pulser sequence duration T
and the emulator time step dt, it reports:
  - the maximum number of qubits that fit in memory,
  - the estimated runtime (optimistic / expected / pessimistic) per qubit count.

Model (from the project report):
  memory   M = (k + 5) * 16 * 2^N                           bytes
  passes   P = diag + k * (23 + 3N) + 5k,  diag = N(N-1)/4 + N + 1
  runtime  r = n_steps * P * 16 * 2^N / (BW * efficiency) * n_traj,  n_steps = T / dt
  k        interpolated from measured values (see K_TABLE)

Example (A100 40 GB):
  python emu_sv_estimator.py --memory-gb 40 --bandwidth-gbs 1555 --duration-ns 3000 --dt-ns 10
"""
import argparse
import math

import matplotlib.pyplot as plt

# dt (ns) -> (k at N=6, growth of k per extra qubit), from the measured Fig. 2.
K_TABLE = {5: (7, 0.0), 10: (8, 0.35), 20: (11, 0.8), 50: (17, 2.0)}
EFFICIENCIES = {"optimistic": 1.0, "expected": 0.5, "pessimistic": 0.25}


def estimate_k(n, dt):
    """Krylov vectors per step. dt is rounded UP to the nearest tabulated value (worst case)."""
    key = next((d for d in sorted(K_TABLE) if d >= dt), max(K_TABLE))
    k0, growth = K_TABLE[key]
    return math.ceil(k0 + growth * max(n - 6, 0))


def memory_bytes(n, dt):
    return (estimate_k(n, dt) + 5) * 16 * 2**n


def max_qubits(capacity_bytes, dt, limit=64):
    best = 0
    for n in range(1, limit + 1):
        if memory_bytes(n, dt) <= capacity_bytes:
            best = n
        else:
            break
    return best


def runtime_seconds(n, dt, duration, bandwidth, efficiency, n_traj):
    k = estimate_k(n, dt)
    passes = n * (n - 1) / 4 + n + 1 + k * (23 + 3 * n) + 5 * k
    steps = math.ceil(duration / dt)  # EMU-SV takes floor(T/dt) steps plus one remainder step
    return steps * passes * 16 * 2**n / (bandwidth * efficiency) * n_traj


def fmt_time(s):
    for unit, size in (("days", 86400), ("h", 3600), ("min", 60)):
        if s >= size:
            return f"{s / size:.1f} {unit}"
    return f"{s:.2f} s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-gb", type=float, required=True, help="memory capacity in GB (GPU memory, or RAM for CPU runs)")
    ap.add_argument("--bandwidth-gbs", type=float, required=True, help="total memory bandwidth in GB/s")
    ap.add_argument("--duration-ns", type=float, required=True, help="T: duration of the Pulser sequence in ns")
    ap.add_argument("--dt-ns", type=float, default=10.0, help="emulator time step in ns (default 10)")
    ap.add_argument("--n-traj", type=int, default=1, help="number of trajectories (default 1)")
    ap.add_argument("--n-qubits", type=int, default=None, help="optional: also report this specific N")
    ap.add_argument("--out", default="emu_sv_estimate.png", help="output image for the graphs")
    ap.add_argument("--show", action="store_true", help="also open the plot window")
    a = ap.parse_args()

    if a.dt_ns <= 0 or a.duration_ns <= 0:
        raise SystemExit("dt and T must be positive.")
    if a.dt_ns > max(K_TABLE):
        print(f"Warning: dt={a.dt_ns} ns is above the measured range (max {max(K_TABLE)} ns); k is likely underestimated.")

    cap, bw = a.memory_gb * 1e9, a.bandwidth_gbs * 1e9
    n_max = max_qubits(cap, a.dt_ns)
    if n_max == 0:
        raise SystemExit("Not even 1 qubit fits in the given memory.")

    def runtimes(n):
        return {name: runtime_seconds(n, a.dt_ns, a.duration_ns, bw, eff, a.n_traj) for name, eff in EFFICIENCIES.items()}

    def describe(n):
        rt = runtimes(n)
        return (f"N={n}: k={estimate_k(n, a.dt_ns)}, peak memory {memory_bytes(n, a.dt_ns) / 1e9:.1f} GB, runtime "
                + ", ".join(f"{name} {fmt_time(t)}" for name, t in rt.items()))

    print(f"Memory {a.memory_gb} GB | bandwidth {a.bandwidth_gbs} GB/s | T={a.duration_ns} ns, dt={a.dt_ns} ns, n_traj={a.n_traj}")
    print(f"Maximum qubits that fit: {n_max}")
    print("  " + describe(n_max))
    if a.n_qubits is not None:
        if a.n_qubits > n_max:
            print(f"Requested N={a.n_qubits} does NOT fit (needs {memory_bytes(a.n_qubits, a.dt_ns) / 1e9:.1f} GB).")
        else:
            print("Requested " + describe(a.n_qubits))

    ns = list(range(1, n_max + 1))
    fig, (ax_mem, ax_rt) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_mem.semilogy(ns, [memory_bytes(n, a.dt_ns) / 1e9 for n in ns], marker="o")
    ax_mem.axhline(a.memory_gb, color="gray", linestyle="--", label=f"capacity ({a.memory_gb:g} GB)")
    ax_mem.set(xlabel="Number of qubits N", ylabel="Peak memory (GB)", title="Memory consumption")
    for name in EFFICIENCIES:
        ax_rt.semilogy(ns, [runtimes(n)[name] for n in ns], marker="o", label=name)
    ax_rt.set(xlabel="Number of qubits N", ylabel="Runtime (s)", title=f"Estimated runtime (T={a.duration_ns:g} ns, dt={a.dt_ns:g} ns)")
    for ax in (ax_mem, ax_rt):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"Graphs saved to {a.out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
