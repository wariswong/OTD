"""
Runopendss_All20032026.py
--------------------------
Recreated helper module for TransferOptimizer.py.

TransferOptimizer.py was copied into this project without the module it
depends on for GeoJSON parsing and OpenDSS conversion — this file did not
exist anywhere on disk (confirmed by an exhaustive search of the repo and
related working folders). Rather than guess at electrical parameters, the
conductor impedance table, transformer template, and phase-designation
mapping below are ported verbatim from optimized_transformer_group_300669.py
(this project's other, working OpenDSS-conversion feature, which processes
the same PEA network JSON schema), so the physics matches what's already
calibrated and in production elsewhere in this app.
"""

import re
import math
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import opendssdirect as odss

# =====================================================================
# GeoJSON-ish (ArcGIS feature) accessors
# Contract inferred from TransferOptimizer.py's call sites: every feature is
# {"attributes": {...}, "geometry": {"x":..,"y":..} | {"paths": [[[x,y],...]]}}
# =====================================================================

def get_attr(f: dict, key: str, default=None):
    return (f.get("attributes") or {}).get(key, default)


def has_point(f: dict) -> bool:
    g = f.get("geometry") or {}
    return ("x" in g) and ("y" in g)


def has_paths(f: dict) -> bool:
    g = f.get("geometry") or {}
    paths = g.get("paths")
    return isinstance(paths, list) and len(paths) > 0


def point_xy(f: dict) -> Tuple[float, float]:
    g = f.get("geometry") or {}
    return (float(g["x"]), float(g["y"]))


def endpoints_from_paths(f: dict) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """First/last vertex of the first path — matches how TransferOptimizer
    treats every LC feature as a single edge between two graph nodes."""
    g = f.get("geometry") or {}
    paths = g.get("paths")
    if not paths or not paths[0]:
        return None
    p = paths[0]
    p1 = (float(p[0][0]), float(p[0][1]))
    p2 = (float(p[-1][0]), float(p[-1][1]))
    return p1, p2


# =====================================================================
# cluster_points — ported verbatim from optimized_transformer_group_300669.py
# =====================================================================

def cluster_points(points: List[Tuple[float, float]], tol: float) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Merge points within `tol` of each other (grid-bucketed 3x3 neighbor search).
    Returns (ids, reps): ids[i] is the cluster id of points[i]; reps[cid] is
    that cluster's representative coordinate (the first point seen for it).
    """
    if tol <= 0:
        raise ValueError("tol must be > 0")

    reps: List[Tuple[float, float]] = []
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    ids: List[int] = []
    tol2 = tol * tol

    for (x, y) in points:
        x = float(x); y = float(y)
        cx = int(round(x / tol))
        cy = int(round(y / tol))

        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in grid.get((cx + dx, cy + dy), []):
                    rx, ry = reps[nid]
                    if (x - rx) ** 2 + (y - ry) ** 2 <= tol2:
                        found = nid
                        break
                if found is not None:
                    break
            if found is not None:
                break

        if found is None:
            nid = len(reps)
            reps.append((x, y))
            grid[(cx, cy)].append(nid)
            found = nid

        ids.append(found)

    return ids, reps


# =====================================================================
# Phase designation + conductor impedance — ported verbatim from
# optimized_transformer_group_300669.py (same PEA network data schema).
# =====================================================================

def _phase_nodes_from_designation(pd) -> List[int]:
    """PHASEDESIGNATION -> OpenDSS phase node list. A=1, B=2, C=3."""
    try:
        pd_int = int(float(pd))
        return {
            7: [1, 2, 3],   # ABC
            6: [1, 2],      # AB
            5: [3, 1],      # CA
            4: [1],         # A
            3: [2, 3],      # BC
            2: [2],         # B
            1: [3],         # C
        }.get(pd_int, [1, 2, 3])
    except Exception:
        pass

    s = (str(pd) if pd is not None else "").strip().upper()
    if not s:
        return [1, 2, 3]
    out = []
    if "A" in s: out.append(1)
    if "B" in s: out.append(2)
    if "C" in s: out.append(3)
    return out if out else [1, 2, 3]


# Aluminium waterproof (AW) 50/95 mm² impedance (Ω/km)
_AW_IMP = {
    50: {"R1": 0.552500, "X1": 0.360233, "R0": 1.153394, "X0": 0.985503},
    95: {"R1": 0.308100, "X1": 0.340069, "R0": 0.643187, "X0": 0.930338},
}


def _impedance_from_size_and_type(conductor_type, conductor_size) -> Tuple[float, float, float, float]:
    """Returns (R1, X1, R0, X0) in Ω/km."""
    ctype = (str(conductor_type) if conductor_type is not None else "").strip().upper()

    size = None
    if isinstance(conductor_size, (int, float)):
        size = int(round(float(conductor_size)))
    else:
        s = (str(conductor_size) if conductor_size is not None else "").strip()
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            try:
                size = int(round(float(m.group(1))))
            except Exception:
                size = None
    if not size or size <= 0:
        size = 25

    is_aw = ("AW" == ctype) or ctype.startswith("AW") or ("AW" in ctype)
    if is_aw and size in _AW_IMP:
        v = _AW_IMP[size]
        return v["R1"], v["X1"], v["R0"], v["X0"]

    # Fallback estimate for sizes outside the calibrated AW table
    base_r1_25 = 0.641
    r1 = base_r1_25 * (25.0 / max(size, 1))
    x1 = 0.083
    return float(r1), float(x1), float(r1 * 3.0), float(x1 * 3.0)


def _sanitize_dss_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "UNKNOWN"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "UNKNOWN"


# =====================================================================
# build_bfs_order
# =====================================================================

def build_bfs_order(n_nodes: int, lc_edges: List[Tuple[int, int]], transformer_node: int):
    """
    BFS from `transformer_node` over `lc_edges` (list of (u, v) node-id pairs).

    Returns (bus_of_node, adj, parent, parent_edge):
      bus_of_node : {node_id: bus_name} — every node 0..n_nodes-1 gets a
                    deterministic "N{id:06d}" bus name, except the
                    transformer's own node which is named "source" (matching
                    the `bus=source.1.2.3.0` secondary winding convention).
      adj         : {node_id: [(neighbor_id, edge_idx), ...]}
      parent      : {node_id: parent_node_id or None} (transformer_node -> None)
      parent_edge : {child_node_id: edge_idx used to reach it from its parent}

    Only bus naming needs to be reproducible across independent calls (once
    from convert_json_to_dss_ordered's own clustering, once from re-parsing
    the same JSON via NetworkGraph) — since both use identical clustering
    inputs/tolerance, the resulting node ids — and therefore bus names —
    line up automatically.
    """
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for idx, (u, v) in enumerate(lc_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    parent: Dict[int, Optional[int]] = {transformer_node: None}
    parent_edge: Dict[int, int] = {}
    queue = deque([transformer_node])
    while queue:
        u = queue.popleft()
        for v, eidx in adj.get(u, []):
            if v in parent:
                continue
            parent[v] = u
            parent_edge[v] = eidx
            queue.append(v)

    bus_of_node = {nid: f"N{nid:06d}" for nid in range(n_nodes)}
    bus_of_node[transformer_node] = "source"

    return bus_of_node, dict(adj), parent, parent_edge


# =====================================================================
# convert_json_to_dss_ordered
# =====================================================================

def convert_json_to_dss_ordered(json_path: str, dss_out_path: str, snap_tol: float = 2.0):
    """
    Parse one network JSON (transformer + LC backbone lines + point loads)
    and write a solvable OpenDSS .dss file. Returns (dss_out_path, bus_of_node).
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    features = data.get("features", [])

    xf_list = [f for f in features if has_point(f) and "XF" in str(get_attr(f, "TAG", "")).upper()]
    if not xf_list:
        raise RuntimeError(f"No transformer (TAG contains XF) in {json_path}")
    xfmr = xf_list[0]
    xfmr_attrs = xfmr.get("attributes") or {}

    lc_list = [(i, f) for i, f in enumerate(features)
               if has_paths(f) and "LC" in str(get_attr(f, "TAG", "")).upper()]
    if not lc_list:
        raise RuntimeError(f"No backbone lines (TAG contains LC) in {json_path}")

    raw_ends: List[Tuple[float, float]] = []
    for _, f in lc_list:
        ep = endpoints_from_paths(f)
        if ep:
            raw_ends.extend([ep[0], ep[1]])

    node_ids, reps = cluster_points(raw_ends, tol=snap_tol)
    reps_arr = np.array(reps, dtype=float)

    def nearest_node(pt: Tuple[float, float]) -> int:
        d2 = (reps_arr[:, 0] - pt[0]) ** 2 + (reps_arr[:, 1] - pt[1]) ** 2
        return int(np.argmin(d2))

    lc_edges: List[Tuple[int, int]] = []
    lc_attrs_by_edge: Dict[int, dict] = {}
    raw_idx = 0
    for feat_idx, f in lc_list:
        ep = endpoints_from_paths(f)
        if not ep:
            continue
        u = node_ids[raw_idx]
        v = node_ids[raw_idx + 1]
        raw_idx += 2
        lc_edges.append((u, v))
        lc_attrs_by_edge[len(lc_edges) - 1] = f.get("attributes") or {}

    transformer_node = nearest_node(point_xy(xfmr))
    bus_of_node, _adj, _parent, _parent_edge = build_bfs_order(len(reps), lc_edges, transformer_node)

    # --- point loads (SUBTYPECODE == 1) ---
    load_recs = []
    for f in features:
        try:
            subtype = int(get_attr(f, "SUBTYPECODE", -1) or -1)
        except Exception:
            subtype = -1
        if subtype != 1 or not has_point(f):
            continue
        tag = str(get_attr(f, "TAG", "") or "").upper()
        if "XF" in tag:
            continue
        peano = str(get_attr(f, "PEANO", "") or "").strip()
        peameter = str(get_attr(f, "PEAMETER", "") or "").strip()
        name = peano or peameter
        if not name:
            continue
        nid = nearest_node(point_xy(f))
        kw = float(get_attr(f, "KWP", 0.0) or 0.0)
        pd = get_attr(f, "PHASEDESIGNATION", 7)
        load_recs.append((name, nid, kw, pd))

    # --- transformer parameters ---
    tx_tag = str(xfmr_attrs.get("TAG", "XFM1") or "XFM1")
    tx_name = _sanitize_dss_name(tx_tag)
    rate_kva = float(xfmr_attrs.get("RATEKVA", 250.0) or 250.0)
    hv_kv = float(xfmr_attrs.get("OPVOLTINT", 22000.0) or 22000.0) / 1000.0
    lv_kv = 0.4
    cfg = str(xfmr_attrs.get("CONFIGURATION", "") or "").strip().upper()
    hv_conn = "Delta" if cfg == "D" else "Wye"

    lines = [
        "Clear",
        "Set DefaultBaseFrequency=50",
        f"New Circuit.{tx_name} basekv={hv_kv:.3f} pu=1.0 phases=3 bus1=SourceBus Angle=0",
        "",
        f"New Transformer.{tx_name} Phases=3 Windings=2 Xhl=2.72 conns=(delta, wye)",
        f"~ wdg=1 bus=SourceBus conn={hv_conn} kv={hv_kv:.3f} kva={rate_kva:.2f} %r=0.5",
        f"~ wdg=2 bus=source.1.2.3.0 conn=wye kv={lv_kv:.3f} kva={rate_kva:.2f} %r=0.5",
        "",
        "! --- LV backbone ---",
    ]

    used_line_names = set()
    for idx, (u, v) in enumerate(lc_edges):
        attrs = lc_attrs_by_edge.get(idx, {})
        tag = str(attrs.get("TAG", "") or f"LC_{idx + 1}")
        name = _sanitize_dss_name(tag)
        if name in used_line_names:
            name = f"{name}_{idx + 1}"
        used_line_names.add(name)

        pd = attrs.get("PHASEDESIGNATION", 7)
        phs = _phase_nodes_from_designation(pd)

        length_m = attrs.get("MEASURELENGTH", None)
        if not isinstance(length_m, (int, float)):
            length_m = attrs.get("SHAPE.LEN", 0.0)
        length_km = float(length_m or 0.0) / 1000.0

        ctype = attrs.get("CONDUCTORTYPE", "")
        csize = attrs.get("CONDUCTORSIZE", 25)
        r1, x1, r0, x0 = _impedance_from_size_and_type(ctype, csize)

        b1, b2 = bus_of_node[u], bus_of_node[v]
        if len(phs) == 3:
            phases, bus1, bus2 = 3, f"{b1}.1.2.3.0", f"{b2}.1.2.3.0"
        elif len(phs) == 2:
            phases, bus1, bus2 = 2, f"{b1}.{phs[0]}.{phs[1]}.0", f"{b2}.{phs[0]}.{phs[1]}.0"
        else:
            phases, bus1, bus2 = 1, f"{b1}.{phs[0]}.0", f"{b2}.{phs[0]}.0"

        lines.append(
            f"New Line.{name} Phases={phases} Bus1={bus1} Bus2={bus2} "
            f"Length={length_km:.6f} Units=km R1={r1:.6f} X1={x1:.6f} R0={r0:.6f} X0={x0:.6f}"
        )

    lines += ["", "! --- Loads ---"]
    kv_ln = lv_kv / math.sqrt(3.0)
    used_load_names = set()
    for name, nid, kw, pd in load_recs:
        load_name = _sanitize_dss_name(name)
        if load_name in used_load_names:
            load_name = f"{load_name}_{nid}"
        used_load_names.add(load_name)

        bus = bus_of_node[nid]
        phs = _phase_nodes_from_designation(pd)
        if len(phs) == 3:
            lines.append(
                f"New Load.{load_name} Bus1={bus}.1.2.3.0 Phases=3 "
                f"kV={lv_kv:.5f} kW={kw:.6f} pf=0.875 Conn=Wye Model=1 Vminpu=0.2"
            )
        else:
            pnode = phs[0]
            lines.append(
                f"New Load.{load_name} Bus1={bus}.{pnode}.0 Phases=1 "
                f"kV={kv_ln:.5f} kW={kw:.6f} pf=0.875 Conn=Wye Model=1 Vminpu=0.2"
            )

    lines += ["", f"Set Voltagebases=[{hv_kv:.3f} {lv_kv:.3f}]", ""]

    with open(dss_out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return str(dss_out_path), bus_of_node


# =====================================================================
# solve_with_opendss
# =====================================================================

def solve_with_opendss(dss_file: str) -> bool:
    """Compile + solve a .dss file on the shared opendssdirect engine.
    Returns whether the solution converged."""
    odss.Basic.ClearAll()
    odss.Text.Command(f'Compile "{dss_file}"')
    odss.Text.Command("CalcVoltageBases")
    odss.Text.Command("Solve")
    return bool(odss.Solution.Converged())
