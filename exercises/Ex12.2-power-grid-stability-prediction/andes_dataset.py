"""
andes_dataset.py — Generate the Ex12.2 stability dataset using ANDES.

Produces the same 180 dispatches × 6 contingencies = 1,080 CCT labels
as problem.py's build_dataset, but computed with ANDES instead of the
hand-written classical model. The dispatches come from problem.py's
operating_points so the inputs are identical — only the labels differ.

Two modes:
    model="classical"  — GENCLS, fair comparison with problem.py
    model="detailed"   — GENROU + ESST1A + TGOV1

Usage on Colab:
    !pip install andes
    # upload problem.py, course_core.py, pinn_core.py, andes_compare.py

    import andes_dataset as ad
    data = ad.build(n_ops=180, model="classical")  # ~1-2 hours
    ad.compare_labels(data)                         # side-by-side table

    # Or start small:
    data = ad.build(n_ops=10, model="classical")   # ~5 min, 60 cases

The dataset is cached to disk after the first run.

Requires: pip install andes
          problem.py (for operating_points and contingencies)
          andes_compare.py (for compute_cct)
"""

import os
import time
import numpy as np

try:
    import problem as pb
    HAS_PROBLEM = True
except ImportError:
    HAS_PROBLEM = False
    print("  [andes_dataset] problem.py not found — needed for operating_points")

try:
    import andes_compare as ac
    HAS_ANDES = ac.HAS_ANDES
except ImportError:
    HAS_ANDES = False
    print("  [andes_dataset] andes_compare.py not found")


CACHE_DIR = "Ex12.2_andes_outputs"


def build(n_ops=180, seed=12, model="classical", cache=True, verbose=True):
    """Generate the full dataset.

    Parameters
    ----------
    n_ops : int
        Number of operating points (dispatches). 180 matches problem.py.
    seed : int
        Seed for operating_points. 12 matches problem.py.
    model : str
        "classical" (GENCLS) or "detailed" (GENROU+AVR+Gov).
    cache : bool
        Save/load from disk to avoid recomputation.

    Returns
    -------
    dict with keys:
        cct        : (n_ops * n_cont,) CCT in seconds
        op_id      : (n_ops * n_cont,) which dispatch
        cont_id    : (n_ops * n_cont,) which contingency
        ops        : (n_ops, 2, 6) the operating points [P; Q]
        model      : str, which ANDES model was used
    """
    if not HAS_PROBLEM:
        raise ImportError("problem.py required for operating_points")
    if not HAS_ANDES:
        raise ImportError("andes not installed — pip install andes")

    # Check cache
    cache_path = os.path.join(CACHE_DIR,
                              f"andes_{model}_{n_ops}ops_seed{seed}.npz")
    if cache and os.path.exists(cache_path):
        if verbose:
            print(f"  loading cached dataset from {cache_path}")
        d = dict(np.load(cache_path, allow_pickle=True))
        d["model"] = str(d.get("model", model))
        return d

    # Generate operating points (same as problem.py)
    ops = pb.operating_points(n_ops, seed=seed)
    cases = ac.contingencies()
    n_cont = len(cases)
    n_total = n_ops * n_cont

    if verbose:
        print(f"  ANDES dataset: {n_ops} dispatches × {n_cont} contingencies"
              f" = {n_total} cases")
        print(f"  model: {model}")
        print(f"  estimated time: {n_total * 3 / 60:.0f}–{n_total * 6 / 60:.0f} min")
        print()

    cct = np.zeros(n_total)
    op_id = np.zeros(n_total, dtype=int)
    cont_id = np.zeros(n_total, dtype=int)

    t0 = time.time()
    failed = 0

    for i in range(n_ops):
        P, Q = ops[i, 0], ops[i, 1]

        for j, c in enumerate(cases):
            idx = i * n_cont + j

            try:
                cct_val = ac.compute_cct(
                    fault_bus=c["fault_bus"],
                    trip_line=c["outage"],
                    model=model,
                    P=P, Q=Q,
                )
            except Exception as e:
                cct_val = np.nan
                failed += 1
                if verbose:
                    print(f"    FAILED op={i} cont={j}: {e}")

            cct[idx] = cct_val
            op_id[idx] = i
            cont_id[idx] = c["index"]

        # Progress
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (n_ops - i - 1)
            done_pct = 100 * (i + 1) / n_ops
            n_done = (i + 1) * n_cont
            insecure = np.sum(cct[:n_done] <= ac.PROTECTION_TIME)
            print(f"  [{done_pct:5.1f}%]  dispatch {i+1}/{n_ops}"
                  f"   {elapsed:.0f}s elapsed   ~{remaining:.0f}s remaining"
                  f"   insecure so far: {insecure}/{n_done}")

    elapsed = time.time() - t0

    if verbose:
        valid = ~np.isnan(cct)
        insecure = np.sum(cct[valid] <= ac.PROTECTION_TIME)
        print(f"\n  completed {n_total} labels in {elapsed:.1f} s"
              f" ({elapsed/n_total*1e3:.0f} ms each)")
        print(f"  failed: {failed}")
        print(f"  CCT  min {np.nanmin(cct)*1e3:.1f}   "
              f"median {np.nanmedian(cct)*1e3:.1f}   "
              f"max {np.nanmax(cct)*1e3:.1f} ms")
        print(f"  insecure at {ac.PROTECTION_TIME*1e3:.0f} ms: "
              f"{insecure} of {valid.sum()} ({100*insecure/valid.sum():.1f}%)")

    data = {
        "cct": cct,
        "op_id": op_id,
        "cont_id": cont_id,
        "ops": ops,
        "model": np.array([model]),
    }

    # Cache
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, **data)
        if verbose:
            print(f"  saved to {cache_path}")

    return data


def compare_labels(andes_data=None, n_ops=180, seed=12, model="classical",
                   verbose=True):
    """Side-by-side comparison of problem.py vs ANDES labels."""

    if not HAS_PROBLEM:
        raise ImportError("problem.py required")

    # Get problem.py labels
    pb_data = pb.build_dataset(n_ops=n_ops, seed=seed)
    cct_pb = pb_data["cct"]

    # Get ANDES labels
    if andes_data is None:
        andes_data = build(n_ops=n_ops, seed=seed, model=model)
    cct_an = andes_data["cct"]

    if len(cct_pb) != len(cct_an):
        print(f"  size mismatch: problem.py {len(cct_pb)} vs ANDES {len(cct_an)}")
        return

    valid = ~np.isnan(cct_an)
    diff = cct_an[valid] - cct_pb[valid]

    if verbose:
        print(f"\n  === Label comparison: problem.py vs ANDES {model} ===")
        print(f"  cases: {valid.sum()} (of {len(cct_an)})")
        print(f"  CCT difference (ANDES − problem.py):")
        print(f"    mean:   {diff.mean()*1e3:+.1f} ms")
        print(f"    median: {np.median(diff)*1e3:+.1f} ms")
        print(f"    std:    {diff.std()*1e3:.1f} ms")
        print(f"    min:    {diff.min()*1e3:+.1f} ms")
        print(f"    max:    {diff.max()*1e3:+.1f} ms")

        insecure_pb = cct_pb[valid] <= ac.PROTECTION_TIME
        insecure_an = cct_an[valid] <= ac.PROTECTION_TIME
        agree = (insecure_pb == insecure_an).sum()
        print(f"\n  Security classification:")
        print(f"    problem.py insecure: {insecure_pb.sum()}"
              f" ({100*insecure_pb.mean():.1f}%)")
        print(f"    ANDES insecure:      {insecure_an.sum()}"
              f" ({100*insecure_an.mean():.1f}%)")
        print(f"    agree:               {agree}"
              f" ({100*agree/valid.sum():.1f}%)")
        print(f"    problem.py secure, ANDES insecure: "
              f"{((~insecure_pb) & insecure_an).sum()}"
              f"  (dangerous disagreement)")
        print(f"    problem.py insecure, ANDES secure: "
              f"{(insecure_pb & (~insecure_an)).sum()}"
              f"  (conservative disagreement)")

    # Per-contingency breakdown
    cases = ac.contingencies()
    if verbose:
        print(f"\n  Per-contingency median CCT:")
        print(f"  {'contingency':<46s}{'problem.py':>12s}{'ANDES':>10s}"
              f"{'diff':>10s}")
        print("  " + "─" * 78)
        for c in cases:
            m = andes_data["cont_id"] == c["index"]
            m_valid = m & valid
            med_pb = np.median(cct_pb[m_valid]) * 1e3
            med_an = np.median(cct_an[m_valid]) * 1e3
            print(f"  {c['label']:<46s}{med_pb:>10.1f} ms{med_an:>9.1f} ms"
                  f"{med_an - med_pb:>+9.1f} ms")

    return {"diff": diff, "cct_pb": cct_pb[valid], "cct_an": cct_an[valid]}


def quick_test(n_ops=5, model="classical", verbose=True):
    """Fast sanity check — 5 dispatches × 6 contingencies = 30 cases."""
    if verbose:
        print("  Quick test: 5 dispatches, classical model\n")
    data = build(n_ops=n_ops, seed=12, model=model, cache=False, verbose=verbose)
    return data


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running quick test (5 dispatches)...\n")
    data = quick_test(n_ops=5)
