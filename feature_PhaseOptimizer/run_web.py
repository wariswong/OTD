"""
run_web.py (feature_PhaseOptimizer)
------------------------------------
Web-callable wrapper for LVOptimizer. Unlike feature_shareload/run_web.py
(invoked as a subprocess), this is designed to be imported and called
in-process from the Flask app — mirroring how optimized_transformer_group_310869
.main_pipeline() is called directly from app/services/project_service.py.

Produces, under <out_dir>:
    results.json               - summary of baseline/steps-applied/final state
    lv_lines.geojson           - LV backbone lines (for the ArcGIS map)
    meter_groups.geojson       - per-meter points (phase before/after, moved?)
    feature_groups.geojson     - transformer / low-voltage point markers
    upgrade_lines.geojson      - conductor-upgrade / phase-addition edges (lines)
    downloads/phase_opt_<facilityid>.xlsx

Usage (CLI):
    python run_web.py <FACILITYID> <OUT_DIR> [REGION]

Usage (import):
    from run_web import run_phase_optimizer
    summary = run_phase_optimizer(facilityid, out_dir, region=region)
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

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PhaseOptimizer_03092026 import (         # noqa: E402
    LVOptimizer, SimResult, save_excel_report, draw_map,
    _meter_current_pd, _build_meter_inventory, PD_TO_PHASE,
)
import pyproj                        # noqa: E402

_UTM47N_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)


def _ll(x, y):
    """UTM47N (x, y) -> [lon, lat] for GeoJSON."""
    lon, lat = _UTM47N_TO_WGS84.transform(x, y)
    return [round(lon, 7), round(lat, 7)]


def _sim_summary(r: "SimResult | None") -> dict | None:
    if r is None:
        return None
    return {
        "converged": r.converged,
        "min_v": round(r.min_v, 1),
        "tr_loading_pct": round(r.tr_loading_pct, 1),
        "phase_imbalance_pct": round(r.phase_imbalance_pct, 1),
        "low_v_count": len(r.low_v_nodes),
        "error": r.error or None,
    }


def _build_results_json(opt: LVOptimizer) -> dict:
    baseline_has_problem = opt._has_problem(opt.baseline) if opt.baseline else None
    final = opt.final_result or opt.baseline

    steps_applied = []
    if opt.phase_moves:
        steps_applied.append("phase_transfer")
    if opt.applied_upgrade:
        steps_applied.append("conductor_upgrade")
    if opt.applied_phase_add:
        steps_applied.append("phase_addition")
    if opt.phase_moves2:
        steps_applied.append("phase_transfer_2")

    final_has_problem = opt._has_problem(final) if final else None
    improved = bool(
        baseline_has_problem and final is not None and final_has_problem is False
    )

    return {
        "facility_id": opt.facilityid,
        "region": opt.region,
        "min_voltage_v": opt.min_voltage_v,
        "max_imbalance_pct": opt.max_imbalance_pct,
        "baseline": _sim_summary(opt.baseline),
        "baseline_has_problem": baseline_has_problem,
        "steps_applied": steps_applied,
        "phase_transfer": {
            "n_moves": len(opt.phase_moves),
            "result": _sim_summary(opt.phase_result),
        } if opt.phase_moves else None,
        "conductor_upgrade": {
            "edge": list(opt.applied_upgrade.edge),
            "from_size": opt.applied_upgrade.from_size,
            "to_size": opt.applied_upgrade.to_size,
            "n_affected": opt.applied_upgrade.n_affected,
            "n_segments": len(opt.applied_upgrade.upgraded_edges),
            "result": _sim_summary(opt.upgrade_result),
        } if opt.applied_upgrade else None,
        "phase_addition": {
            "n_edges_upgraded": len(opt.applied_phase_add.upgraded_edges),
            "n_meters_moved": len(opt.applied_phase_add.meter_moves),
            "result": _sim_summary(opt.applied_phase_add.result),
        } if opt.applied_phase_add else None,
        "final": _sim_summary(final),
        "final_has_problem": final_has_problem,
        "improved": improved,
    }


def _write_geojson_layers(opt: LVOptimizer, out_dir: Path) -> list:
    """Write lv_lines/meter_groups/feature_groups/upgrade_lines geojson.

    Returns the list of meters whose phase actually changed (peano, kw,
    phase_before, phase_after) so the caller can surface it as a readable
    activity list on the results page, not just as rings on the map.
    """
    net = opt.net
    baseline_raw = opt.raw
    final_raw = opt.final_raw or opt.raw
    moved_meters = []

    # ---- lv_lines.geojson: backbone LV lines from the parsed network graph ----
    line_features = []
    for u, v in net.G.edges():
        pu, pv = net.node_coords.get(u), net.node_coords.get(v)
        if not pu or not pv:
            continue
        line_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "paths": None,
                         "coordinates": [_ll(*pu), _ll(*pv)]},
            "properties": {"u": u, "v": v},
        })
    with open(out_dir / "lv_lines.geojson", "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": line_features}, fh, ensure_ascii=False)

    # ---- meter_groups.geojson: one point per meter, phase before/after ----
    meters_before = {m.feat_idx: m for m in _build_meter_inventory(net, baseline_raw)}
    meter_features = []
    for feat_idx, node_id in net.load_feat_nodes.items():
        before = meters_before.get(feat_idx)
        if before is None:
            continue
        pt = net.node_coords.get(node_id)
        if not pt:
            continue
        after_feat = final_raw["features"][feat_idx] if feat_idx < len(final_raw["features"]) else None
        after_pd = _meter_current_pd(after_feat) if after_feat is not None else before.current_pd
        phase_before = PD_TO_PHASE.get(before.current_pd, "?")
        phase_after = PD_TO_PHASE.get(after_pd, "?")
        moved = after_pd != before.current_pd
        meter_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": _ll(*pt)},
            "properties": {
                "peano": before.peano,
                "kw": round(before.kw, 2),
                "phase_before": phase_before,
                "phase_after": phase_after,
                "moved": 1 if moved else 0,
                # คู่เฟส "ก่อน->หลัง" เช่น "A->B" ใช้แยกสีวงแหวนตามทิศทางการย้ายเฟสบนแผนที่
                "phase_move": f"{phase_before}->{phase_after}",
            },
        })
        if moved:
            moved_meters.append({
                "peano": before.peano or f"nid={node_id}",
                "kw": round(before.kw, 2),
                "phase_before": phase_before,
                "phase_after": phase_after,
            })
    moved_meters.sort(key=lambda m: (m["phase_before"], m["phase_after"], m["peano"]))
    with open(out_dir / "meter_groups.geojson", "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": meter_features}, fh, ensure_ascii=False)

    # ---- feature_groups.geojson: transformer / low-voltage point markers ----
    feature_groups = []

    def _add_point(node_id, name, group):
        pt = net.node_coords.get(node_id)
        if not pt:
            return
        feature_groups.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": _ll(*pt)},
            "properties": {"name": name, "group": group},
        })

    _add_point(net.transformer_node, f"TR: {opt.facilityid}", "transformer")

    baseline_low_v = set(opt.baseline.low_v_nodes) if opt.baseline else set()
    final_low_v = set((opt.final_result or opt.baseline).low_v_nodes) if opt.baseline else set()
    for node_id in sorted(baseline_low_v - final_low_v):
        _add_point(node_id, f"แก้ไขแรงดันตกแล้ว (node {node_id})", "low_v_fixed")
    for node_id in sorted(final_low_v):
        _add_point(node_id, f"แรงดันตกยังคงอยู่ (node {node_id})", "low_v_remaining")

    with open(out_dir / "feature_groups.geojson", "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feature_groups}, fh, ensure_ascii=False)

    # ---- upgrade_lines.geojson: conductor-upgrade / phase-addition edges ----
    upgrade_lines = []

    def _add_line(u, v, name, group):
        pu, pv = net.node_coords.get(u), net.node_coords.get(v)
        if not (pu and pv):
            return
        upgrade_lines.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [_ll(*pu), _ll(*pv)]},
            "properties": {"name": name, "group": group},
        })

    if opt.applied_upgrade:
        # อัปเกรดทั้ง path (ทุก segment จากหม้อแปลง→โหนดแรงดันต่ำ) ไม่ใช่แค่ entry
        # edge เส้นเดียว — ต้องวาดทุก segment ใน upgraded_edges ให้ตรงกับที่
        # draw_map()/draw_interactive_map() ใน PhaseOptimizer_03092026.py แสดง
        # ไม่งั้นแผนที่เว็บจะโชว์แค่ช่วงแรกช่วงเดียว สั้นกว่าที่อัปเกรดจริง
        name = f"เพิ่มขนาดสาย →{opt.applied_upgrade.to_size}mm² ({len(opt.applied_upgrade.upgraded_edges)} ช่วง)"
        for _fi, eu, ev in opt.applied_upgrade.upgraded_edges:
            _add_line(eu, ev, name, "conductor_upgrade")

    if opt.applied_phase_add:
        for _fi, eu, ev in opt.applied_phase_add.upgraded_edges:
            _add_line(eu, ev, "เพิ่มเฟสสาย → 3 เฟส", "phase_addition")

    with open(out_dir / "upgrade_lines.geojson", "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": upgrade_lines}, fh, ensure_ascii=False)

    return moved_meters


def run_phase_optimizer(
    facilityid: str,
    out_dir: str,
    region: str | None = None,
    min_voltage_v: float = 200.0,
    max_imbalance_pct: float = 25.0,
    max_sim_time_s: float = 300.0,
) -> dict:
    """Run LVOptimizer end-to-end and write all web-facing output files.

    Returns the same dict written to results.json (so the caller — the
    Flask route — can respond immediately without re-reading the file).
    Raises on failure (JSON fetch error, baseline not converging, etc.) so
    the route can surface a clear error instead of a silently-empty result.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "downloads").mkdir(parents=True, exist_ok=True)

    optimizer = LVOptimizer(
        facilityid=facilityid,
        min_voltage_v=min_voltage_v,
        max_imbalance_pct=max_imbalance_pct,
        max_sim_time_s=max_sim_time_s,
        region=region,
    )
    optimizer.run()

    if optimizer.net is None or optimizer.baseline is None:
        raise RuntimeError(optimizer.error or f"วิเคราะห์ {facilityid} ไม่สำเร็จ — ตรวจสอบ JSON / GIS connection")
    if not optimizer.baseline.converged:
        raise RuntimeError(optimizer.error or f"Baseline ไม่ converge: {optimizer.baseline.error}")

    save_excel_report(optimizer, str(out_path / "downloads" / f"phase_opt_{facilityid}.xlsx"))
    draw_map(optimizer, str(out_path / "downloads" / f"phase_opt_{facilityid}.png"))
    moved_meters = _write_geojson_layers(optimizer, out_path)

    results = _build_results_json(optimizer)
    results["moved_meters"] = moved_meters
    with open(out_path / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    return results


def main() -> None:
    if len(sys.argv) < 3:
        print("ERROR: Usage: python run_web.py <FACILITYID> <OUT_DIR> [REGION]", flush=True)
        sys.exit(1)
    facilityid = sys.argv[1].strip()
    out_dir = sys.argv[2]
    region = sys.argv[3].strip() if len(sys.argv) > 3 and sys.argv[3].strip() else None

    try:
        results = run_phase_optimizer(facilityid, out_dir, region=region)
    except Exception:
        logging.exception(f"[phase_optimizer] {facilityid} failed")
        sys.exit(1)

    logging.info(f"[phase_optimizer] {facilityid} done: {results['final']}")


if __name__ == "__main__":
    main()
