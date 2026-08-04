import os, json, math, re
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import argparse

import numpy as np
import csv
import opendssdirect as dss

fac_raw: str = ""


# ---------------------------
# Helpers
# ---------------------------

def get_attr(f: Dict[str, Any], key: str, default=None):
    return (f.get("attributes") or {}).get(key, default)

def has_point(f: Dict[str, Any]) -> bool:
    g = f.get("geometry") or {}
    return ("x" in g) and ("y" in g)

def has_paths(f: Dict[str, Any]) -> bool:
    g = f.get("geometry") or {}
    return isinstance(g.get("paths"), list) and len(g.get("paths")) > 0

def point_xy(f: Dict[str, Any]) -> Tuple[float, float]:
    g = f.get("geometry") or {}
    return float(g["x"]), float(g["y"])

def endpoints_from_paths(f: Dict[str, Any]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    g = f.get("geometry") or {}
    paths = g.get("paths")
    if not paths: return None
    p = paths[0]
    if not p: return None
    return (float(p[0][0]), float(p[0][1])), (float(p[-1][0]), float(p[-1][1]))

def sanitize_min(name: str) -> str:
    name = (name or "").strip()
    if not name: return "UNKNOWN"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "UNKNOWN"

def build_meter_bus_names(features: List[Dict[str, Any]]) -> Dict[int, str]:
    """Map each load-point feature's index in `features` to the OpenDSS meter
    bus name convert_json_to_dss_ordered() assigns it ("M_<name>").

    Replicates that function's load-point filter and duplicate-name
    suffixing exactly (convert_json_to_dss_ordered calls this same helper),
    so other code — e.g. voltage lookups matching a meter back to its
    simulated bus — has one source of truth instead of re-deriving the
    naming/dedup rules and risking silent drift from what actually got
    written to the .dss file.
    """
    names: Dict[int, str] = {}
    used_load_names: set = set()
    for i, f in enumerate(features):
        if not has_point(f):
            continue
        try:
            subtype = int(get_attr(f, "SUBTYPECODE", -999) or -999)
        except Exception:
            subtype = -999
        if subtype != 1:
            continue
        tag = str(get_attr(f, "TAG", "") or "").upper()
        if "XF" in tag:
            continue
        peano = str(get_attr(f, "PEANO", "") or "").strip()
        peameter = str(get_attr(f, "PEAMETER", "") or "").strip()
        if not peano and not peameter:
            continue
        if not peano:
            peano = str(get_attr(f, "TAG", "") or get_attr(f, "PEAMETER", "") or "UNKNOWN").strip()
        load_name = sanitize_min(peano)
        if load_name in used_load_names:
            suffix = 2
            while f"{load_name}_{suffix}" in used_load_names:
                suffix += 1
            load_name = f"{load_name}_{suffix}"
        used_load_names.add(load_name)
        names[i] = f"M_{load_name}"
    return names

def phase_nodes_from_designation(pd: Any) -> List[int]:
    # 7-ABC, 6-AB, 5-CA, 4-A, 3-BC, 2-B, 1-C
    try:
        pd = int(pd)
    except Exception:
        pd = 7
    return {
        7: [1, 2, 3],
        6: [1, 2],
        5: [3, 1],
        4: [1],
        3: [2, 3],
        2: [2],
        1: [3],
    }.get(pd, [1, 2, 3])

def phase_node_from_letter(letter: Any) -> Optional[int]:
    s = (str(letter) if letter is not None else "").strip().upper()
    return {"A": 1, "B": 2, "C": 3}.get(s)

def cluster_points(points: List[Tuple[float, float]], tol: float):
    reps: List[Tuple[float, float]] = []
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    ids: List[int] = []
    tol2 = tol * tol
    for (x, y) in points:
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
                if found is not None: break
            if found is not None: break
        if found is None:
            nid = len(reps)
            reps.append((x, y))
            grid[(cx, cy)].append(nid)
            found = nid
        ids.append(found)
    return ids, reps


# ---------------------------
# Impedance (Ω/km)
# ---------------------------

AW_IMP = {
    50: {"R1": 0.552500, "X1": 0.360233, "R0": 1.153394, "X0": 0.985503},
    95: {"R1": 0.308100, "X1": 0.340069, "R0": 0.643187, "X0": 0.930338},
}

def impedance_from_json(conductor_type: Any, conductor_size: Any):
    ctype = (str(conductor_type) if conductor_type is not None else "").strip().upper()
    try:
        size = int(float(conductor_size))
    except Exception:
        size = 25

    if ctype == "AW" and size in AW_IMP:
        v = AW_IMP[size]
        return v["R1"], v["X1"], v["R0"], v["X0"]

    # fallback simple scaling — conductor ไม่อยู่ในตาราง AW_IMP
    print(f"[Warn] impedance_from_json: conductor type='{ctype}' size={size}mm2 ไม่อยู่ในตาราง — ใช้ค่า fallback scaling (ค่าโดยประมาณ)")
    base_r1_25 = 0.641
    r1 = base_r1_25 * (25.0 / max(size, 1))
    x1 = 0.083
    r0 = r1 * 3.0
    x0 = x1 * 3.0
    return r1, x1, r0, x0


# ---------------------------
# Core: Build graph, BFS order, orient lines
# ---------------------------

def build_bfs_order(num_nodes: int, edges: List[Tuple[int, int]], source: int):
    adj = [[] for _ in range(num_nodes)]
    for ei, (u, v) in enumerate(edges):
        adj[u].append((v, ei))
        adj[v].append((u, ei))

    parent = [-1] * num_nodes
    level = [-1] * num_nodes
    parent_edge = [-1] * num_nodes

    q = deque([source])
    level[source] = 0
    parent[source] = source

    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v, ei in adj[u]:
            if level[v] != -1:
                continue
            level[v] = level[u] + 1
            parent[v] = u
            parent_edge[v] = ei
            q.append(v)

    # สร้าง mapping: node_id -> new bus name (เรียงจาก source)
    new_bus = {}
    for idx, nid in enumerate(order):
        new_bus[nid] = f"S{idx:06d}"
    new_bus[source] = "source"  # ให้ source ชื่อเดิมตาม pattern

    # อาจมี node ที่ disconnected (island): ใส่ชื่อท้ายๆ
    rest = [i for i in range(num_nodes) if i not in new_bus]
    base = len(order)
    for k, nid in enumerate(rest):
        new_bus[nid] = f"I{base + k:06d}"

    tree_edge_set = set(e for e in parent_edge if e != -1)

    return new_bus, parent, level, tree_edge_set


# ---------------------------
# Convert JSON -> DSS (ordered)
# ---------------------------
def Lossdocument():
    return [
        {'rating_kVA':  30, 'no_load_loss': 0.12, 'load_loss':  0.430},
        {'rating_kVA':  50, 'no_load_loss': 0.11, 'load_loss':  0.875},
        {'rating_kVA': 100, 'no_load_loss': 0.15, 'load_loss':  1.450},
        {'rating_kVA': 160, 'no_load_loss': 0.26, 'load_loss':  2.000},
        {'rating_kVA': 250, 'no_load_loss': 0.36, 'load_loss':  2.750},
        {'rating_kVA': 315, 'no_load_loss': 0.44, 'load_loss':  3.250},
        {'rating_kVA': 400, 'no_load_loss': 0.52, 'load_loss':  3.850},
        {'rating_kVA': 500, 'no_load_loss': 0.61, 'load_loss':  4.600},
    ]

def lookup_tx_losses_kW(rating_kva: float):
    tab = Lossdocument()
    best = min(tab, key=lambda r: abs(float(r['rating_kVA']) - float(rating_kva)))
    return float(best["no_load_loss"]), float(best["load_loss"]), float(best["rating_kVA"])

def convert_json_to_dss_ordered(
    json_path: str,
    dss_out: str,
    snap_tol: float = 2.0,
    svc_match: float = 20.0,
    basekv_hv: float = 22.0,
    basekv_lv: float = 0.4,
    default_pf: float = 0.875,
    
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])

    xf_list = [f for f in features if has_point(f) and "XF" in str(get_attr(f, "TAG", "")).upper()]
    if not xf_list:
        raise RuntimeError("ไม่พบหม้อแปลง: TAG มี 'XF'")
    
    FAC_RE = re.compile(r"^\d{2}-\d{6}$")

    # เลือก XF ที่ FACILITYID 
    xfmr = None
    for f in xf_list:
        fac = str(get_attr(f, "FACILITYID", "") or "").strip()
        if FAC_RE.match(fac):
            xfmr = f
            break

    # ถ้าไม่มีแบบ NN-NNNNNN ค่อย fallback ตัวแรก
    if xfmr is None:
        xfmr = xf_list[0]


    # Backbone LC (line)
    lc_lines = [f for f in features if has_paths(f) and "LC" in str(get_attr(f, "TAG", "")).upper()]
    if not lc_lines:
        raise RuntimeError("ไม่พบสาย backbone: TAG มี 'LC'")

    # Loads (SUBTYPE=1 point) - exclude transformer point
    load_points = []
    for f in features:
        if not has_point(f):
            continue

        try:
            subtype = int(get_attr(f, "SUBTYPECODE", -999) or -999)
        except Exception:
            subtype = -999

        if subtype != 1:
            continue

        tag = str(get_attr(f, "TAG", "") or "").upper()

        # กันไม่ให้หม้อแปลง XF ถูกสร้างเป็น Load
        if "XF" in tag:
            continue

        # เอาเฉพาะ meter จริง
        peano = str(get_attr(f, "PEANO", "") or "").strip()
        peameter = str(get_attr(f, "PEAMETER", "") or "").strip()
        if not peano and not peameter:
            continue

        load_points.append(f)

    if not load_points:
        raise RuntimeError("ไม่พบ load meters: SUBTYPECODE=1 (point) หลังจากกรอง XF")

    # Service (SUBTYPE=5 line)
    service_lines = []
    for f in features:
        if not has_paths(f):
            continue
        try:
            subtype = int(get_attr(f, "SUBTYPECODE", -999) or -999)
        except Exception:
            subtype = -999
        if subtype == 5:
            service_lines.append(f)

    # Build node clusters from LC endpoints
    endpoints = []
    rec = []  # (feature, node_u, node_v) later
    raw_ends = []
    for f in lc_lines:
        ep = endpoints_from_paths(f)
        if not ep:
            continue
        raw_ends.extend([ep[0], ep[1]])
    node_ids, reps = cluster_points(raw_ends, tol=snap_tol)
    if not reps:
        raise RuntimeError("ไม่พบ endpoint ที่ valid ใน LC lines ทั้งเส้น — ตรวจสอบ geometry ของสาย backbone")

    # Map each LC line to node ids
    # node_ids list is [u0,v0,u1,v1,...] in the same order as raw_ends appended
    # We need to rebuild in same sequence:
    idx = 0
    edges = []
    lc_records = []
    for f in lc_lines:
        ep = endpoints_from_paths(f)
        if not ep:
            continue
        u = node_ids[idx]; v = node_ids[idx + 1]
        idx += 2
        edges.append((u, v))
        lc_records.append((f, u, v))

    reps_arr = np.array(reps, dtype=float).reshape(-1, 2)

    def nearest_node(pt: Tuple[float, float]) -> Tuple[int, float]:
        x, y = pt
        d2 = (reps_arr[:, 0] - x) ** 2 + (reps_arr[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        return i, float(math.sqrt(d2[i]))

    # Source node from transformer location
    source_node, _ = nearest_node(point_xy(xfmr))

    # BFS order + new bus names + spanning tree edges
    new_bus, parent, level, tree_edge_set = build_bfs_order(len(reps), edges, source_node)

    # Transformer params
    # ใช้ FacilityID เป็นชื่อหม้อแปลง (fallback ไป TAG ถ้าไม่มี)
    fac = str(get_attr(xfmr, "FACILITYID", "") or "").strip()
    if not fac:
        fac = str(get_attr(xfmr, "TAG", "XFM1") or "XFM1").strip()

    tx_name = sanitize_min(fac) or "XFM1"
    tx_name_raw = fac  # เก็บไว้โชว์ในรายงานได้

    rate_kva = float(get_attr(xfmr, "RATEKVA", 250.0) or 250.0)
    # --- Loss from table using RATEKVA ---
    core_kW, cu_full_kW, matched_kVA = lookup_tx_losses_kW(rate_kva)

    # %NoLoadLoss = 100 * core_kW / kVA
    pct_noloadloss = 100.0 * (core_kW / rate_kva) if rate_kva > 0 else 0.0

    # Copper loss at full-load in OpenDSS ~ ( (%R1 + %R2)/100 ) * kVA  (kW)
    # -> set %R1 = %R2 = 100 * cu_full_kW / (2*kVA)
    pct_r_each = 100.0 * (cu_full_kW / (2.0 * rate_kva)) if rate_kva > 0 else 0.0

    hv_kv = float((get_attr(xfmr, "OPVOLTINT", 22000.0) or 22000.0)) / 1000.0
    cfg = str(get_attr(xfmr, "CONFIGURATION", "") or "").strip().upper()
    hv_conn = "Delta" if cfg == "D" else "Wye"

    # Meter meta (Load name = PEANO ตรง ๆ) — bus names come from the shared
    # build_meter_bus_names() helper (same filter/order as load_points above)
    # so voltage-lookup code elsewhere can match a meter to its simulated bus
    # without re-deriving this naming/dedup logic separately.
    meter_meta = []
    meter_pts = []
    meter_bus_names = list(build_meter_bus_names(features).values())
    for f, meter_bus in zip(load_points, meter_bus_names):
        load_name = meter_bus[len("M_"):]

        pnode = phase_node_from_letter(get_attr(f, "PHASE"))
        if pnode is None:
            phs = phase_nodes_from_designation(get_attr(f, "PHASEDESIGNATION"))
            pnode = phs[0] if phs else 1

        kw = get_attr(f, "KWP", 0.0)
        kw = float(kw) if isinstance(kw, (int, float)) else 0.0

        pt = point_xy(f)
        meter_meta.append({"load_name": load_name, "meter_bus": meter_bus, "pnode": int(pnode), "kw": kw, "pt": pt})
        meter_pts.append(pt)

    meter_pts_arr = np.array(meter_pts, dtype=float)

    def nearest_meter(pt: Tuple[float, float]) -> Tuple[int, float]:
        x, y = pt
        d2 = (meter_pts_arr[:, 0] - x) ** 2 + (meter_pts_arr[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        return i, float(math.sqrt(d2[i]))

    # --- Pick first 2 LC lines that touch source_node as Feeder1/Feeder2 (by JSON order) ---
    # Single-feeder transformers (only 1 LC line from source) are valid — treat all as Feeder1.
    incident = []  # list of (ei, neighbor_node)
    for ei, (f, u, v) in enumerate(lc_records):
        if u == source_node:
            incident.append((ei, v))
        elif v == source_node:
            incident.append((ei, u))

    if len(incident) == 0:
        raise RuntimeError("source node ไม่มีสาย LC ออกเลย — ตรวจสอบข้อมูล JSON")

    feeder1_ei, feeder1_neighbor = incident[0]
    feeder2_ei, feeder2_neighbor = incident[1] if len(incident) >= 2 else (None, None)

    # --- Build feeder_id for every node using BFS parent chain ---
    # feeder_id[n] = 1/2 if node is in Feeder1/Feeder2, 0 if island/unassigned
    feeder_id = [0] * len(parent)
    feeder_id[source_node] = 0
    feeder_id[feeder1_neighbor] = 1
    if feeder2_neighbor is not None:
        feeder_id[feeder2_neighbor] = 2

    def feeder_of_node(n: int) -> int:
        """walk up parents until reaching source; return 1/2/0"""
        if feeder_id[n] in (1, 2):
            return feeder_id[n]
        if level[n] < 0:
            return 0  # disconnected/island

        cur = n
        # climb until we reach one of first-hop nodes or source
        while cur != source_node and cur != parent[cur]:
            if cur == feeder1_neighbor:
                feeder_id[n] = 1
                return 1
            if feeder2_neighbor is not None and cur == feeder2_neighbor:
                feeder_id[n] = 2
                return 2
            cur = parent[cur]

        # if still not decided, assign from ancestor if exists
        feeder_id[n] = feeder_id[cur] if feeder_id[cur] in (1, 2) else 0
        return feeder_id[n]

    for n in range(len(parent)):
        if n == source_node:
            continue
        if level[n] >= 0:
            feeder_of_node(n)

    # optional debug
    feeder2_tag = get_attr(lc_records[feeder2_ei][0], "TAG") if feeder2_ei is not None else "(none)"
    print("Feeder head lines:",
        get_attr(lc_records[feeder1_ei][0], "TAG"), "-> Feeder1",
        feeder2_tag, "-> Feeder2")


    # Build ordered LC lines: orient from source outward
    used_line_names = set()
    line_cmds = []
    for ei, (f, u, v) in enumerate(lc_records):
        tag = str(get_attr(f, "TAG", "") or f"LC_{ei+1}")
        lname_base = sanitize_min(tag) or f"LC_{ei+1}"

        #  ใส่ feeder suffix ให้ "ทุกเส้น" ตาม feeder ของปลายที่ลึกกว่า (child side)
        deep = u if level[u] >= level[v] else v
        fid = feeder_id[deep]  # 1 / 2 / 0

        suffix = "_Feeder1" if fid == 1 else "_Feeder2" if fid == 2 else "_Island"
        lname = f"{lname_base}{suffix}"


        if lname in used_line_names:
            lname = f"{lname}_{ei+1}"
        used_line_names.add(lname)

        # Decide direction:
        if ei in tree_edge_set:
            # Tree edge: one endpoint must be parent of the other in BFS
            # choose bus1=parent, bus2=child
            if parent[u] == v:
                p, c = v, u
            elif parent[v] == u:
                p, c = u, v
            else:
                # fallback to level
                p, c = (u, v) if level[u] <= level[v] else (v, u)
        else:
            # Non-tree edge (loop): orient by BFS level
            p, c = (u, v) if level[u] <= level[v] else (v, u)

        b1 = new_bus[p]
        b2 = new_bus[c]

        pd = get_attr(f, "PHASEDESIGNATION", 7)
        phs = phase_nodes_from_designation(pd)

        length_m = get_attr(f, "MEASURELENGTH", None)
        if not isinstance(length_m, (int, float)):
            length_m = get_attr(f, "SHAPE.LEN", 0.0)
        length_km = float(length_m or 0.0) / 1000.0

        ctype = get_attr(f, "CONDUCTORTYPE", "")
        csize = get_attr(f, "CONDUCTORSIZE", 25)
        r1, x1, r0, x0 = impedance_from_json(ctype, csize)

        if len(phs) == 3:
            phases = 3
            bus1 = f"{b1}.1.2.3.0"
            bus2 = f"{b2}.1.2.3.0"
        elif len(phs) == 2:
            phases = 2
            bus1 = f"{b1}.{phs[0]}.{phs[1]}.0"
            bus2 = f"{b2}.{phs[0]}.{phs[1]}.0"
        else:
            phases = 1
            bus1 = f"{b1}.{phs[0]}.0"
            bus2 = f"{b2}.{phs[0]}.0"

        line_cmds.append(
            f"New Line.{lname} Phases={phases} Bus1={bus1} Bus2={bus2} "
            f"Length={length_km:.6f} Units=km R1={r1:.6f} X1={x1:.6f} R0={r0:.6f} X0={x0:.6f}"
        )

    # Service lines: connect from nearest backbone node (using new bus names) to meter bus
    svc_cmds = []
    used_svc = set()
    for j, f in enumerate(service_lines, start=1):
        ep = endpoints_from_paths(f)
        if not ep:
            continue
        p1, p2 = ep

        m1, d1 = nearest_meter(p1)
        m2, d2 = nearest_meter(p2)

        if d1 <= d2 and d1 <= svc_match:
            meter_idx, meter_end_is_p1 = m1, True
        elif d2 < d1 and d2 <= svc_match:
            meter_idx, meter_end_is_p1 = m2, False
        else:
            continue

        m = meter_meta[meter_idx]
        meter_bus, pnode = m["meter_bus"], m["pnode"]

        net_pt = p2 if meter_end_is_p1 else p1
        net_node, _ = nearest_node(net_pt)
        net_bus = new_bus[net_node]

        tag = str(get_attr(f, "TAG", "") or f"SVC_{j}")
        sname = sanitize_min(tag) or f"SVC_{j}"
        if sname in used_svc:
            sname = f"{sname}_{j}"
        used_svc.add(sname)

        length_m = get_attr(f, "MEASURELENGTH", None)
        if not isinstance(length_m, (int, float)):
            length_m = get_attr(f, "SHAPE.LEN", 0.0)
        length_km = float(length_m or 0.0) / 1000.0

        # ค่า service line ตามที่คุณเคยขอ (ถ้าจะบังคับให้เล็กมาก)
        # r1 = x1 = r0 = x0 = 0.0001
        r1, x1, r0, x0 = 1.2, 0.08, 3.6, 0.24

        svc_cmds.append(
            f"New Line.{sname} Phases=1 Bus1={net_bus}.{pnode}.0 Bus2={meter_bus}.{pnode}.0 "
            f"Length={length_km:.6f} Units=km R1={r1:.6f} X1={x1:.6f} R0={r0:.6f} X0={x0:.6f}"
        )

    # Loads
    kv_ln = basekv_lv / math.sqrt(3.0)
    load_cmds = []
    kw_total = 0.0
    for m in meter_meta:
        kw_total += float(m["kw"])
        load_cmds.append(
            f"New Load.{m['load_name']} Bus1={m['meter_bus']}.{m['pnode']}.0 Phases=1 "
            f"kV={kv_ln:.5f} kW={m['kw']:.6f} pf={default_pf:.3f} Conn=Wye Model=1 Vminpu=0.2"
        )
    
    # Write DSS
    os.makedirs(os.path.dirname(os.path.abspath(dss_out)), exist_ok=True)
    with open(dss_out, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "Clear",
            "Set DefaultBaseFrequency=50",
            f"new circuit.ckt basekv={basekv_hv:.3f} pu=1.0 phases=3 bus1=SourceBus Angle=0",
            "",
            f"! ordered by BFS from transformer -> source node={source_node}",
            f"! LC lines={len(line_cmds)}  service matched={len(svc_cmds)}  loads={len(load_cmds)}  sumkW={kw_total:.3f}",
            "",
            f"New Transformer.{tx_name} Phases=3 Windings=2 Xhl=2.72 conns=(delta, wye) %noloadloss={pct_noloadloss:.6f}",
            f"~ wdg=1 bus=SourceBus conn={hv_conn} kv={hv_kv:.3f} kva={rate_kva:.2f} %r={pct_r_each:.6f}",
            f"~ wdg=2 bus=source.1.2.3.0 conn=wye kv={basekv_lv:.3f} kva={rate_kva:.2f} %r={pct_r_each:.6f}",
            "",
            "! --- LV backbone (LC...) oriented from source outward ---",
            *line_cmds,
            "",
            "! --- Service drops (SUBTYPE=5) ---",
            *svc_cmds,
            "",
            "! --- Loads ---",
            *load_cmds,
            "",
             f"Set Voltagebases=[{basekv_hv:.3f} {basekv_lv:.3f}]",
             "CalcVoltageBases",
             "Solve",
            # "Show Voltages LN Nodes",
            # "Show Powers",
            # "Show Currents",
            # "",
        ]) + "\n")

    return os.path.abspath(dss_out), tx_name_raw  

# ---------------------------
# OpenDSS Solve + Query results
# ---------------------------


def solve_with_opendss(dss_file: str):
    # OpenDSS's "Compile" command silently chdir()s the whole process into the
    # compiled file's directory (so relative paths inside the .dss resolve).
    # It never restores it, so any relative path resolved after this call
    # (e.g. TransferOptimizer opening the *next* facility's cached JSON by a
    # relative path) breaks with FileNotFoundError. Save/restore cwd around it.
    _cwd = os.getcwd()
    try:
        dss.Basic.ClearAll()
        dss.Command(f'Compile "{os.path.abspath(dss_file)}"')
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print("[Warn] Solution not converged")
    finally:
        os.chdir(_cwd)

def _bus_base(bus_with_nodes: str) -> str:
    return bus_with_nodes.split(".")[0].strip()

def _nodes_from_bus_str(bus_with_nodes: str):
    # "M_xxx.1.0" -> [1,0]
    parts = bus_with_nodes.split(".")
    nodes = []
    for p in parts[1:]:
        try:
            nodes.append(int(p))
        except:
            pass
    return nodes

def bus_vmag_v(busname: str):
    dss.Circuit.SetActiveBus(busname)
    nodes = dss.Bus.Nodes()              # e.g. [1,2,3,0]
    vmag = dss.Bus.VMagAngle()[0::2]     # V
    return dict(zip(nodes, vmag))

def bus_vmag_pu(busname: str):
    dss.Circuit.SetActiveBus(busname)
    nodes = dss.Bus.Nodes()
    vpu = dss.Bus.puVmagAngle()[0::2]    # pu
    return dict(zip(nodes, vpu))

def element_pq_kwkvar(elem: str):
    dss.Circuit.SetActiveElement(elem)
    pq = dss.CktElement.Powers()         # kW,kvar
    return list(zip(pq[0::2], pq[1::2]))

def element_currents_a(elem: str):
    dss.Circuit.SetActiveElement(elem)
    ia = dss.CktElement.CurrentsMagAng() # A,deg
    return list(zip(ia[0::2], ia[1::2]))

def element_losses_kw(elem: str):
    dss.Circuit.SetActiveElement(elem)
    w, var = dss.CktElement.Losses()     # W,var
    return w/1000.0, var/1000.0

def _sum_pq_terminal(pq_pairs, terminal=1, phases=3):
    """
    pq_pairs: [(P1,Q1),(P2,Q2)...] ตามลำดับ conductor/terminal
    terminal=1 -> เอา 3 เฟสแรก (หรือ phases แรก)
    terminal=2 -> ข้าม phases ตัวแรกไป
    """
    start = (terminal-1)*phases
    end = start + phases
    P = sum(pq_pairs[i][0] for i in range(start, min(end, len(pq_pairs))))
    Q = sum(pq_pairs[i][1] for i in range(start, min(end, len(pq_pairs))))
    return P, Q

def _phase_label(nodes: list[int]) -> str:
    # map node->phase label
    m = {1:"A",2:"B",3:"C"}
    ph = [m[n] for n in nodes if n in m]
    return "".join(ph) if ph else "?"
def element_terminal_current_dict(elem: str):
    """
    คืน dict กระแสของ element แบบแยก terminal:
    out[terminal_index][node] = I_mag (A)
    terminal_index เริ่มที่ 1
    """
    dss.Circuit.SetActiveElement(elem)
    magsangs = dss.CktElement.CurrentsMagAng()  # [I1,ang1,I2,ang2,...]
    nodeorder = dss.CktElement.NodeOrder()      # [node1,node2,...] ความยาว = ncond*nterm
    ncond = dss.CktElement.NumConductors()
    nterm = dss.CktElement.NumTerminals()

    # currents per conductor index
    I_mag = magsangs[0::2]

    out = {}
    for t in range(nterm):
        tdict = {}
        base = t * ncond
        for k in range(ncond):
            idx = base + k
            if idx >= len(nodeorder) or idx >= len(I_mag):
                continue
            node = nodeorder[idx]
            if node == 0:
                continue
            tdict[int(node)] = float(I_mag[idx])
        out[t + 1] = tdict
    return out
def tx_secondary_power_by_node(tx_elem: str):
    """
    คืน dict node -> (P,Q) ของ terminal 2 (secondary) หน่วย kW/kvar
    ใช้ NodeOrder จับคู่ให้ตรงจริง ๆ
    """
    dss.Circuit.SetActiveElement(tx_elem)

    pq = dss.CktElement.Powers()             # [P1,Q1,P2,Q2,...]
    pq_pairs = list(zip(pq[0::2], pq[1::2]))

    nodeorder = dss.CktElement.NodeOrder()   # ยาว = ncond*nterm (เรียงตาม Powers/Currents)
    ncond = dss.CktElement.NumConductors()

    # terminal 2 (secondary)
    base = 1 * ncond

    out = {}
    for k in range(ncond):
        idx = base + k
        if idx >= len(nodeorder) or idx >= len(pq_pairs):
            continue
        node = int(nodeorder[idx])
        if node == 0:        # ไม่เอา neutral
            continue
        out[node] = (float(pq_pairs[idx][0]), float(pq_pairs[idx][1]))
    return out


def tx_secondary_power_abc_and_sum(tx_elem: str):
    m = tx_secondary_power_by_node(tx_elem)
    PA, QA = m.get(1, (0.0, 0.0))
    PB, QB = m.get(2, (0.0, 0.0))
    PC, QC = m.get(3, (0.0, 0.0))
    Psum = PA + PB + PC
    Qsum = QA + QB + QC
    return (PA, QA, PB, QB, PC, QC, Psum, Qsum)

def tx_secondary_currents_abc(tx_elem: str):
    dss.Circuit.SetActiveElement(tx_elem)

    magsangs = dss.CktElement.CurrentsMagAng()
    I_mag = magsangs[0::2]

    nodeorder = dss.CktElement.NodeOrder()
    ncond = dss.CktElement.NumConductors()

    # terminal2
    base = 1 * ncond

    m = {}
    for k in range(ncond):
        idx = base + k
        if idx < len(nodeorder) and idx < len(I_mag):
            node = int(nodeorder[idx])
            if node != 0:
                m[node] = float(I_mag[idx])

    return m.get(1), m.get(2), m.get(3)  # IA,IB,IC
def percent_unbalance(values):
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < 2:
        return None
    avg = sum(vals) / len(vals)
    if avg <= 0:
        return None
    dev = max(abs(v - avg) for v in vals)
    return 100.0 * dev / avg


def get_bus_voltage(busname: str):
    dss.Circuit.SetActiveBus(busname)
    nodes = dss.Bus.Nodes()
    vmag = dss.Bus.VMagAngle()[0::2]
    #vpu  = dss.Bus.puVmagAngle()[0::2]
    return nodes, dict(zip(nodes, vmag))

def collect_selected_buses():
    """
    เลือก bus เฉพาะ:
    - bus ที่มี Load ต่ออยู่
    - bus ปลายสายของ Line ทุกเส้น (Bus1/Bus2)
    คืนเป็น set ของ bus_base (ตัด .1.2.3.0 ออกแล้ว)
    """
    buses = set()

    # จาก Loads: ใช้ bus ของ element load (ชัวร์กว่า Loads.BusName)
    for lname in dss.Loads.AllNames():
        elem = f"Load.{lname}"
        dss.Circuit.SetActiveElement(elem)
        b = dss.CktElement.BusNames()
        if b:
            buses.add(b[0].split(".")[0])

    # จาก Lines: เอา Bus1/Bus2
    for e in dss.Circuit.AllElementNames():
        if not e.lower().startswith("line."):
            continue
        dss.Circuit.SetActiveElement(e)
        b = dss.CktElement.BusNames()
        if len(b) >= 1 and b[0]:
            buses.add(b[0].split(".")[0])
        if len(b) >= 2 and b[1]:
            buses.add(b[1].split(".")[0])

    return buses
def element_terminal_pq_by_node(elem: str):
    """
    out[terminal_index][node] = (P_kW, Q_kvar)
    """
    dss.Circuit.SetActiveElement(elem)
    pq = dss.CktElement.Powers()
    pq_pairs = list(zip(pq[0::2], pq[1::2]))

    nodeorder = dss.CktElement.NodeOrder()
    ncond = dss.CktElement.NumConductors()
    nterm = dss.CktElement.NumTerminals()

    out = {}
    for t in range(nterm):
        base = t * ncond
        m = {}
        for k in range(ncond):
            idx = base + k
            if idx >= len(nodeorder) or idx >= len(pq_pairs):
                continue
            node = int(nodeorder[idx])
            if node == 0:
                continue
            m[node] = (float(pq_pairs[idx][0]), float(pq_pairs[idx][1]))
        out[t + 1] = m
    return out

def element_terminal_i_by_node(elem: str):
    """
    out[terminal_index][node] = I_mag(A)
    """
    dss.Circuit.SetActiveElement(elem)
    magsangs = dss.CktElement.CurrentsMagAng()
    I_mag = magsangs[0::2]

    nodeorder = dss.CktElement.NodeOrder()
    ncond = dss.CktElement.NumConductors()
    nterm = dss.CktElement.NumTerminals()

    out = {}
    for t in range(nterm):
        base = t * ncond
        m = {}
        for k in range(ncond):
            idx = base + k
            if idx >= len(nodeorder) or idx >= len(I_mag):
                continue
            node = int(nodeorder[idx])
            if node == 0:
                continue
            m[node] = float(I_mag[idx])
        out[t + 1] = m
    return out

def export_node_voltage_selected_csv(out_csv: str, run_id: str = "", facilityid: str = ""):
    buses = sorted(collect_selected_buses())

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["RunID", "FACILITYID", "Bus", "Node", "Vmag_V"])

        for b in buses:
            nodes, vV = get_bus_voltage(b)
            for n in nodes:
                w.writerow([run_id, facilityid, b, n, vV.get(n)])

def build_line_label_map_from_dss(dss_file: str) -> dict:
    """
    อ่านไฟล์ .dss ที่ generate แล้วดึงชื่อ Line.<name> ตามที่เขียนไว้ในไฟล์ (คง case เดิม)
    คืน dict: key=lowercase object name, value=original-cased object name
    เช่น {"2233lc000004169_feeder1": "2233LC000004169_Feeder1"}
    """
    mp = {}
    pat = re.compile(r"^\s*new\s+line\.([^\s]+)", re.IGNORECASE)
    with open(dss_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            name_in_file = m.group(1).strip().strip('"').strip("'")
            mp[name_in_file.lower()] = name_in_file
    return mp

def get_line_names_from_dss(dss_file: str) -> list[str]:
    names = []
    pat = re.compile(r"^\s*new\s+line\.([^\s]+)", re.IGNORECASE)
    with open(dss_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line)
            if m:
                names.append(m.group(1).strip().strip('"').strip("'"))
    return names


def export_line_end_metrics_csv(out_csv: str, dss_file: str, run_id: str = "", facilityid: str = ""):
    line_names = get_line_names_from_dss(dss_file)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "RunID", "FACILITYID",
            "Line", "Bus1", "Bus2",
            "Loss_kW", "Loss_kvar",
            "V1_A_V", "V1_B_V", "V1_C_V",
            "V2_A_V", "V2_B_V", "V2_C_V",
            "P1_A_kW", "Q1_A_kvar", "P1_B_kW", "Q1_B_kvar", "P1_C_kW", "Q1_C_kvar",
            "P2_A_kW", "Q2_A_kvar", "P2_B_kW", "Q2_B_kvar", "P2_C_kW", "Q2_C_kvar",
            "I1_A_A", "I1_B_A", "I1_C_A",
            "I2_A_A", "I2_B_A", "I2_C_A",
        ])

        for lname in line_names:
            elem = f"Line.{lname}"
            try:
                dss.Circuit.SetActiveElement(elem)
            except:
                continue

            buses = dss.CktElement.BusNames()
            bus1 = buses[0] if len(buses) > 0 else ""
            bus2 = buses[1] if len(buses) > 1 else ""

            w_loss, var_loss = dss.CktElement.Losses()
            loss_kw, loss_kvar = w_loss/1000.0, var_loss/1000.0

            b1 = bus1.split(".")[0] if bus1 else ""
            b2 = bus2.split(".")[0] if bus2 else ""

            v1A = v1B = v1C = v2A = v2B = v2C = None
            if b1:
                _, vV1 = get_bus_voltage(b1)
                v1A, v1B, v1C = vV1.get(1), vV1.get(2), vV1.get(3)
            if b2:
                _, vV2 = get_bus_voltage(b2)
                v2A, v2B, v2C = vV2.get(1), vV2.get(2), vV2.get(3)

            pqT = element_terminal_pq_by_node(elem)
            iT  = element_terminal_i_by_node(elem)

            (P1A, Q1A) = pqT.get(1, {}).get(1, (None, None))
            (P1B, Q1B) = pqT.get(1, {}).get(2, (None, None))
            (P1C, Q1C) = pqT.get(1, {}).get(3, (None, None))
            I1A = iT.get(1, {}).get(1)
            I1B = iT.get(1, {}).get(2)
            I1C = iT.get(1, {}).get(3)

            (P2A, Q2A) = pqT.get(2, {}).get(1, (None, None))
            (P2B, Q2B) = pqT.get(2, {}).get(2, (None, None))
            (P2C, Q2C) = pqT.get(2, {}).get(3, (None, None))
            I2A = iT.get(2, {}).get(1)
            I2B = iT.get(2, {}).get(2)
            I2C = iT.get(2, {}).get(3)

            w.writerow([
                run_id, facilityid,
                lname, bus1, bus2,
                loss_kw, loss_kvar,
                v1A, v1B, v1C,
                v2A, v2B, v2C,
                P1A, Q1A, P1B, Q1B, P1C, Q1C,
                P2A, Q2A, P2B, Q2B, P2C, Q2C,
                I1A, I1B, I1C,
                I2A, I2B, I2C,
            ])


def report_transformer_pattern():
    tx_names = []
    try:
        tx_names = dss.Transformers.AllNames()
    except Exception:
        tx_names = []

    if not tx_names:
        # fallback debug: ดูว่ามี element อะไรบ้าง
        elems = dss.Circuit.AllElementNames()
        raise RuntimeError(
            "ไม่พบ Transformer ใน circuit (Transformers.AllNames ว่าง)\n"
            f"ตัวอย่าง elements 30 ตัวแรก: {elems[:30]}"
        )

    tx = f"Transformer.{tx_names[0]}"

    dss.Circuit.SetActiveElement(tx)
    buses = dss.CktElement.BusNames()  # [HVbus..., LVbus...]
    hv_bus = buses[0]
    lv_bus = buses[1]

    hv_base = _bus_base(hv_bus)
    lv_base = _bus_base(lv_bus)

    # แรงดันฝั่ง secondary (ใช้ magnitude ที่ bus LV ในหน่วย V และ pu)
    v_lv_v = bus_vmag_v(lv_base)
    v_lv_pu = bus_vmag_pu(lv_base)

    # Power output -> ใช้ terminal 2 (ฝั่ง secondary) ของ transformer
    PA, QA, PB, QB, PC, QC, P2, Q2 = tx_secondary_power_abc_and_sum(tx)

    # Current A,B,C -> terminal 2 เช่นกัน (รองรับ 4 สาย)
    Id = element_terminal_current_dict(tx)
    I2 = Id.get(2, {})  # terminal 2 = secondary
    IA = I2.get(1)      # node 1 = A
    IB = I2.get(2)      # node 2 = B
    IC = I2.get(3)      # node 3 = C

    # Voltage A,B,C ของฝั่ง secondary (node 1,2,3)
    VA = v_lv_v.get(1)
    VB = v_lv_v.get(2)
    VC = v_lv_v.get(3)
    
    # Unbalance ของกระแส (ถ้ามีครบ 3 เฟส)
    I_unb = percent_unbalance([IA, IB, IC])

    # Loss
    loss_kw, loss_kvar = element_losses_kw(tx)

    return {
        "TransformerName": tx,
        "SecondaryBus": lv_bus,
        "SecondaryV(pu)": v_lv_pu,
        "PowerOutput(kW,kvar)_secTerm": (P2, Q2),
        "PowerA_out(kW,kvar)": (PA, QA),
        "PowerB_out(kW,kvar)": (PB, QB),
        "PowerC_out(kW,kvar)": (PC, QC),
        "VoltageA(V)": VA, "VoltageB(V)": VB, "VoltageC(V)": VC,
        "CurrentA(A)": IA, "CurrentB(A)": IB, "CurrentC(A)": IC,
        "CurrentUnbalance_%": I_unb,
        "Loss(kW,kvar)": (loss_kw, loss_kvar),
    }

def report_loads_pattern(limit: int = 0):
    rows = []
    for lname in dss.Loads.AllNames():
        dss.Loads.Name(lname)

        elem = f"Load.{lname}"
        dss.Circuit.SetActiveElement(elem)

        # ✅ เอา bus จาก element แทน Loads.BusName()
        buses = dss.CktElement.BusNames()
        bus = buses[0] if buses else ""
        bus_base = bus.split(".")[0]

        # phase node จาก bus string เช่น "M_xxx.1.0" -> node=1
        parts = bus.split(".")
        ph_node = 1
        if len(parts) >= 2:
            try:
                ph_node = int(parts[1])
            except:
                ph_node = 1
        ph = {1: "A", 2: "B", 3: "C"}.get(ph_node, "?")

        # Load (ค่าที่ตั้งไว้)
        kw = float(dss.Loads.kW())
        kvar = float(dss.Loads.kvar())

        # Voltage ที่ node ของโหลด (V และ pu)
        vV = bus_vmag_v(bus_base).get(ph_node)
        vpu = bus_vmag_pu(bus_base).get(ph_node)

        rows.append({
            "PEANO": lname,                 # ชื่อหลัง New Load.
            "Phase": ph,
            "Load(kW,kvar)": (kw, kvar),
            "Voltage(V,pu)": (vV, vpu),
            "Bus": bus,
        })

        if limit and len(rows) >= limit:
            break
    return rows

def derive_workdir_from_json(json_path: str, output_root: Optional[str] = None) -> str:
    json_path = os.path.abspath(json_path)
    base_dir = output_root or os.path.dirname(json_path)
    stem = os.path.splitext(os.path.basename(json_path))[0]

    m = re.search(r"(\d{2})-(\d{6})", stem)
    if m:
        case_name = f"case_{m.group(1)}_{m.group(2)}"
    else:
        stem2 = re.sub(r"_with_MV$", "", stem, flags=re.IGNORECASE)
        stem2 = sanitize_min(stem2)
        case_name = f"case_{stem2}"

    return os.path.join(base_dir, case_name)

def build_run_id(facilityid: str, json_path: str) -> str:
    stem = os.path.splitext(os.path.basename(json_path))[0]
    fac = sanitize_min(facilityid or "RUN")
    return f"RUN_{fac}_{sanitize_min(stem)}"

# -------------------------
# 1) Transformer CSV
# -------------------------
def export_transformer_csv(tx_csv: str, tx: dict, fac_raw: str, run_id: str):
    with open(tx_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "RunID",
            "FACILITYID",
            "SecondaryBus",
            "PowerOutput_kW", "PowerOutput_kvar",
            "PA_kW","QA_kvar","PB_kW","QB_kvar","PC_kW","QC_kvar",
            "VA_V", "VB_V", "VC_V",
            "IA_A", "IB_A", "IC_A",
            "CurrentUnbalance_%",
            "Loss_kW", "Loss_kvar"
        ])

        P2, Q2 = tx["PowerOutput(kW,kvar)_secTerm"]
        loss_kw, loss_kvar = tx["Loss(kW,kvar)"]

        w.writerow([
            run_id,
            fac_raw,
            tx["SecondaryBus"],
            _fmt(P2), _fmt(Q2),
            _fmt(tx["PowerA_out(kW,kvar)"][0]), _fmt(tx["PowerA_out(kW,kvar)"][1]),
            _fmt(tx["PowerB_out(kW,kvar)"][0]), _fmt(tx["PowerB_out(kW,kvar)"][1]),
            _fmt(tx["PowerC_out(kW,kvar)"][0]), _fmt(tx["PowerC_out(kW,kvar)"][1]),
            _fmt(tx["VoltageA(V)"]), _fmt(tx["VoltageB(V)"]), _fmt(tx["VoltageC(V)"]),
            _fmt(tx["CurrentA(A)"]), _fmt(tx["CurrentB(A)"]), _fmt(tx["CurrentC(A)"]),
            _fmt(tx["CurrentUnbalance_%"]),
            _fmt(loss_kw), _fmt(loss_kvar),
        ])

# -------------------------
# 2) Loads CSV
# -------------------------
def export_loads_csv(loads_csv: str, loads: list, fac_raw: str, run_id: str):
    with open(loads_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["RunID", "FACILITYID", "PEANO", "Phase", "Load_kW", "Load_kvar", "Voltage_V", "Voltage_pu", "Bus"])

        for r in loads:
            kw, kvar = r["Load(kW,kvar)"]
            vV, vpu = r["Voltage(V,pu)"]
            w.writerow([
                run_id,
                fac_raw,
                r["PEANO"],
                r["Phase"],
                _fmt(kw), _fmt(kvar),
                _fmt(vV), _fmt(vpu),
                r.get("Bus", "")
            ])
def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON LV network to DSS, solve, and export CSV reports.")
    parser.add_argument("input_path", nargs="?", default=None,
                        help="Path to one JSON file or a folder containing many JSON files")
    parser.add_argument("--json-path", dest="json_path", default=None,
                        help="Path to one JSON file (explicit form)")
    parser.add_argument("--json-dir", default=None,
                        help="Folder containing many JSON files to process in batch")
    parser.add_argument("--output-root", default=None,
                        help="Base folder for generated case folders; default = same folder as each JSON")
    parser.add_argument("--workdir", default=None,
                        help="Override full working directory directly (single-file mode only)")
    parser.add_argument("--pattern", default="*.json",
                        help="Filename pattern for batch mode, default = *.json")
    parser.add_argument("--recursive", action="store_true",
                        help="Search JSON files recursively in batch mode")
    return parser.parse_args()            

def process_one_json(json_path: str, output_root: Optional[str] = None, workdir: Optional[str] = None):
    global fac_raw

    json_path = os.path.abspath(json_path)

    if workdir:
        workdir = os.path.abspath(workdir)
    else:
        workdir = derive_workdir_from_json(json_path, output_root)

    os.makedirs(workdir, exist_ok=True)

    json_stem = os.path.splitext(os.path.basename(json_path))[0]

    # แยกเก็บ DSS ไว้อีกโฟลเดอร์
    dss_output_dir = r"D:\testpy\dssoutput"
    os.makedirs(dss_output_dir, exist_ok=True)

    dss_out = os.path.join(dss_output_dir, f"{json_stem}.dss")

    out, fac_raw = convert_json_to_dss_ordered(json_path, dss_out)
    run_id = build_run_id(fac_raw, json_path)

    print("JSON_PATH:", json_path)
    print("WORKDIR:", workdir)
    print("DSS_OUT:", out)
    print("Facility Raw:", fac_raw)
    print("RunID:", run_id)

    solve_with_opendss(out)

    node_csv = os.path.join(workdir, "node_voltage_selected.csv")
    line_csv = os.path.join(workdir, "line_end_metrics.csv")
    tx_csv = os.path.join(workdir, "transformer_report.csv")
    loads_csv = os.path.join(workdir, "loads_report.csv")

    export_node_voltage_selected_csv(node_csv, run_id=run_id, facilityid=fac_raw)
    export_line_end_metrics_csv(line_csv, out, run_id=run_id, facilityid=fac_raw)

    tx = report_transformer_pattern()
    loads = report_loads_pattern(limit=0)

    export_transformer_csv(tx_csv, tx, fac_raw, run_id)
    export_loads_csv(loads_csv, loads, fac_raw, run_id)

    print("Wrote DSS:", out)
    print("Wrote:", node_csv)
    print("Wrote:", line_csv)
    print("Wrote:", tx_csv)
    print("Wrote:", loads_csv)
    print("-" * 60)

    return {
        "json_path": json_path,
        "workdir": workdir,
        "dss_out": out,
        "facilityid": fac_raw,
        "run_id": run_id,
        "node_csv": node_csv,
        "line_csv": line_csv,
        "tx_csv": tx_csv,
        "loads_csv": loads_csv,
    }

def iter_json_files(json_dir: str, pattern: str = "*.json", recursive: bool = False):
    json_dir = os.path.abspath(json_dir)
    if recursive:
        files = sorted(Path(json_dir).rglob(pattern))
    else:
        files = sorted(Path(json_dir).glob(pattern))
    return [str(p) for p in files if p.is_file()]

def main():
    args = parse_args()

    input_path = args.input_path
    json_path = args.json_path
    json_dir = args.json_dir

    # positional arg: แยกว่าเป็น file หรือ folder
    if input_path and not json_path and not json_dir:
        if os.path.isdir(input_path):
            json_dir = input_path
        else:
            json_path = input_path

    # ไม่มี arg ใดเลย → ใช้ default batch folder
    if not json_path and not json_dir:
        json_dir = r"D:\testpy\jsonfile"

    if json_path and json_dir:
        raise SystemExit("ใช้ได้ทีละแบบ: ระบุ json_path หรือ json_dir อย่างใดอย่างหนึ่ง")

    if json_path:
        process_one_json(
            json_path=json_path,
            output_root=args.output_root,
            workdir=args.workdir,
        )
        return

    if args.workdir:
        print("หมายเหตุ: --workdir จะถูกใช้เฉพาะ single-file mode เท่านั้น; batch mode จะใช้ derive_workdir_from_json()")

    files = iter_json_files(json_dir, pattern=args.pattern, recursive=args.recursive)
    if not files:
        raise SystemExit(f"ไม่พบไฟล์ JSON ในโฟลเดอร์: {json_dir}")

    print(f"Found {len(files)} JSON file(s)")

    ok = 0
    failed = 0
    failures = []

    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing: {fp}")
        try:
            process_one_json(
                json_path=fp,
                output_root=args.output_root,
                workdir=None,
            )
            ok += 1
        except Exception as e:
            failed += 1
            failures.append((fp, str(e)))
            print(f"FAILED: {fp}")
            print(f"Reason: {e}")
            print("-" * 60)

    print(f"Batch finished. Success={ok}, Failed={failed}")
    if failures:
        print("Failed files:")
        for fp, msg in failures:
            print(f" - {fp} :: {msg}")

def _fmt(x):
    return "" if x is None else (f"{x:.6f}" if isinstance(x, (int, float)) else str(x))


if __name__ == "__main__":
    main()