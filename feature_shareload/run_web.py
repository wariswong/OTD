"""
run_web.py
----------
Web-callable wrapper for TransferOptimizer.
Saves output to a specified directory instead of the hardcoded path in run_transfer.py.

Usage:
    python run_web.py <FAC_A> <FAC_B> <OUT_DIR>

Output files:
    <OUT_DIR>/transfer_<FAC_A>_<FAC_B>.html
    <OUT_DIR>/transfer_<FAC_A>_<FAC_B>.png
    <OUT_DIR>/transfer_<FAC_A>_<FAC_B>.xlsx
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _clean_results_for_json(results, fac_a, fac_b, rated_a, rated_b):
    """Project each scenario dict down to JSON-safe, table-ready fields.

    The raw scenario dicts carry a `subtree_nodes` set (not JSON-serializable)
    and numpy floats from OpenDSS queries, so this rebuilds a plain-Python
    projection instead of dumping the dicts directly.
    """
    scenarios = []
    for i, r in enumerate(results, start=1):
        su, sv = r.get("switch_edge", (None, None))
        a_after = r.get("a_loading_after")
        b_after = r.get("b_loading_after")
        min_v = r.get("min_v")
        scenarios.append({
            "rank": i,
            "switch_u": su,
            "switch_v": sv,
            "tie_node_a": r.get("tie_node_a"),
            "tie_node_b": r.get("tie_node_b"),
            "tie_distance_m": round(float(r.get("tie_distance_m", 0.0)), 1),
            "tie_conductor_size": r.get("tie_conductor_size"),
            "subtree_kw": round(float(r.get("subtree_kw", 0.0)), 1),
            "a_loading_before": round(float(r.get("a_loading_before", 0.0)), 1),
            "a_loading_after": round(float(a_after), 1) if a_after is not None else None,
            "b_loading_before": round(float(r.get("b_loading_before", 0.0)), 1),
            "b_loading_after": round(float(b_after), 1) if b_after is not None else None,
            "min_v": round(float(min_v), 0) if min_v is not None else None,
            "feasible": bool(r.get("feasible")),
            "optimal": bool(r.get("optimal")),
            "error": r.get("error"),
        })
    return {
        "fac_a": fac_a, "fac_b": fac_b,
        "a_rated_kva": round(float(rated_a), 0),
        "b_rated_kva": round(float(rated_b), 0),
        "scenarios": scenarios,
    }


def main():
    if len(sys.argv) < 4:
        print("ERROR: Usage: python run_web.py <FAC_A> <FAC_B> <OUT_DIR>", flush=True)
        sys.exit(1)

    fac_a = sys.argv[1].strip()
    fac_b = sys.argv[2].strip()
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_key = f"{fac_a}_{fac_b}"
    stem = str(out_dir / f"transfer_{pair_key}")

    logging.info(f"Starting TransferOptimizer for {fac_a} <-> {fac_b}")
    logging.info(f"Output dir: {out_dir}")

    from TransferOptimizer import (
        TransferOptimizer,
        save_excel_report,
        draw_topology_map,
        draw_interactive_map,
        collect_optimal_node_voltages,
    )

    optimizer = TransferOptimizer(fac_a=fac_a, fac_b=fac_b, force_refresh=True)
    results = optimizer.run()

    if not results:
        print("NO_RESULTS", flush=True)
        logging.warning("No results from optimizer")
        sys.exit(0)

    logging.info(f"Got {len(results)} results, saving Excel...")
    save_excel_report(results, optimizer.net_a, optimizer.net_b, stem + ".xlsx")

    logging.info("Saving results table JSON...")
    table_data = _clean_results_for_json(
        results, fac_a, fac_b, optimizer.net_a.rated_kva, optimizer.net_b.rated_kva
    )
    with open(stem + "_results.json", "w", encoding="utf-8") as fh:
        json.dump(table_data, fh, ensure_ascii=False)

    volt_a, volt_b, phase_a, phase_b = {}, {}, {}, {}
    raw_a = raw_b = None
    opt = next((r for r in results if r.get("optimal")), None)

    if opt:
        logging.info("Computing voltage map for optimal scenario...")
        with open(optimizer._json_a, encoding="utf-8") as fh:
            raw_a = json.load(fh)
        with open(optimizer._json_b, encoding="utf-8") as fh:
            raw_b = json.load(fh)
        volt_a, volt_b, phase_a, phase_b = collect_optimal_node_voltages(
            optimizer.net_a, optimizer.net_b, raw_a, raw_b, opt,
            snap_tol_a=optimizer.net_a.snap_tol,
            snap_tol_b=optimizer.net_b.snap_tol,
        )
        logging.info(f"Voltage nodes — TR_A: {len(volt_a)}, TR_B: {len(volt_b)}")

    logging.info("Drawing PNG topology map...")
    draw_topology_map(optimizer.net_a, optimizer.net_b, results,
                      stem + ".png", volt_a=volt_a, volt_b=volt_b)

    logging.info("Drawing interactive HTML map...")
    draw_interactive_map(optimizer.net_a, optimizer.net_b, results,
                         stem + ".html",
                         volt_a=volt_a, volt_b=volt_b,
                         phase_a=phase_a, phase_b=phase_b,
                         raw_a=raw_a, raw_b=raw_b)

    print(f"SUCCESS:{stem}.html", flush=True)
    logging.info(f"Done. Output: {stem}.html")


if __name__ == "__main__":
    main()
