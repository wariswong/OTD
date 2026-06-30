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
