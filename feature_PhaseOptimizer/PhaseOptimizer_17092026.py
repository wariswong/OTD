"""
PhaseOptimizer_17092026.py
---------------------------
วิเคราะห์ระบบจำหน่ายแรงต่ำ (LV) ของหม้อแปลงตัวเดียว
และแนะนำการปรับปรุง 3 ขั้นตอน:
  1. ย้ายเฟสมิเตอร์   (Phase Transfer)  — ย้ายให้สอดคล้องกับสายที่มีเฟสรองรับ
  2. เพิ่มขนาดสาย     (Conductor Upgrade) — 50 → 95 mm² (เฉพาะเมื่อมีมิเตอร์ EV)
  3. เพิ่มเฟสสาย      (Phase Addition)   — single/dual-phase → 3-phase พร้อมกระจายโหลด (ไม่เกินเฟสที่หม้อแปลงมี)

Pipeline:
  1. โหลด/ดึง JSON network จาก FacilityID
  2. รัน OpenDSS baseline
  3. ตรวจแรงดันตก / phase imbalance
  4. ถ้าพบปัญหา → ย้ายเฟสมิเตอร์
       4a. ไล่สมดุลจากไลน์ branch ลึกสุดขึ้นมาถึงไลน์เมน (โยกตัวโหลดเยอะสุดก่อน)
       4b. greedy + OpenDSS เก็บงานโหนดแรงดันต่ำที่เหลือ
  5. ถ้ายังไม่ผ่าน → simulate เพิ่มขนาดสาย (เฉพาะเมื่อมี EV ในพื้นที่แรงดันต่ำ)
  6. ถ้ายังไม่ผ่าน → simulate เพิ่มเฟสสาย

Usage:
    python PhaseOptimizer_17092026.py <FACILITYID> [options]

    --min-voltage   float  แรงดันต่ำสุด (V)          [default 200]
    --max-imbalance float  phase imbalance สูงสุด (%) [default 25] — ต่ำกว่านี้ถือว่าผ่าน หยุดย้าย
    --json-dir      path   folder JSON               [default D:\\testpy\\jsonfile]
    --min-move-gain float  ย้าย 1 ตัวต้องลด RMS ต่อเฟสได้ ≥ ค่านี้ (kW) [default 0.02]
    --out-dir       path   folder ผลลัพธ์            [default D:\\TRneighborhood\\output]
"""

import os, sys, json, math, re, argparse, tempfile, shutil, contextlib, io, copy, warnings, time, threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# This file lives in feature_PhaseOptimizer/, but the modules it depends on
# live in feature_shareload/ (Runopendss_All16092026.py, TransferOptimizer_
# 16092026.py) and the project root (InputJsonApi.py). Put both on sys.path so
# imports resolve regardless of the caller's cwd.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_THIS_DIR.parent)
_SHARELOAD_DIR = str(_THIS_DIR.parent / "feature_shareload")
for _p in (_PROJECT_ROOT, _SHARELOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import openpyxl
from openpyxl.styles import PatternFill, Font
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.font_manager as _fm

# ตั้งฟอนต์ไทย — ลองตามลำดับความนิยม
_THAI_FONT_PREF = ["TH SarabunPSK", "TH Sarabun New", "Tahoma", "Leelawadee UI", "Leelawadee"]
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_thai_font = next((f for f in _THAI_FONT_PREF if f in _available_fonts), None)
if _thai_font:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_thai_font, "DejaVu Sans", "Arial"]

# ปิด warning "Glyph X missing from font" — matplotlib ยัง fallback ไปฟอนต์ถัดไปได้เอง
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

from Runopendss_All16092026 import (
    get_attr, has_point, has_paths, point_xy, endpoints_from_paths,
    cluster_points, convert_json_to_dss_ordered, solve_with_opendss, build_bfs_order,
    AW_IMP,
)

# TransferOptimizer_16092026.py's filename changes every promotion round, so it
# can't be reached with a plain `import` statement — load it by file path (same
# trick used in feature_shareload/run_web.py). Using this file (not the older
# TransferOptimizer.py) keeps PhaseOptimizer on the same NetworkGraph/region-
# aware-fetch/multi-phase-transformer modeling as the shareload feature.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "TransferOptimizer_16092026", _THIS_DIR.parent / "feature_shareload" / "TransferOptimizer_16092026.py"
)
_topt_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_topt_mod)

NetworkGraph                 = _topt_mod.NetworkGraph
_collect_dss_bus_phase_voltages = _topt_mod._collect_dss_bus_phase_voltages
query_transformer_loading    = _topt_mod.query_transformer_loading
import opendssdirect as odss

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
JSON_DIR    = r"D:\testpy\jsonfile"
DEFAULT_OUT = r"D:\TRneighborhood\output_phase"

# %Unbalance ที่ถือว่า "ผ่าน" — ต่ำกว่านี้หยุดย้ายมิเตอร์ (ไม่ไล่เกลี่ยต่อจนเป็น 0%)
DEFAULT_MAX_IMBALANCE_PCT = 25.0

# โหลดรายเฟสของหม้อแปลงสูงสุดที่ยอมรับ (% ของพิกัดเฟส PHASEA_KVA / PHASEB_KVA / PHASEC_KVA)
# TR loading เดิมคิดจาก kVA รวม / พิกัดรวม จึงมองไม่เห็นเฟสที่เกินพิกัด —
# ข้อมูลจริง 600 หม้อแปลง: 14 ตัวมีเฟสเกิน 100% โดย 12 ตัวในนั้น TR loading รวมยังไม่ถึง 100%
DEFAULT_MAX_PHASE_LOADING_PCT = 100.0

_PHASE_KVA_FIELDS = {1: "PHASEA_KVA", 2: "PHASEB_KVA", 3: "PHASEC_KVA"}

PHASE_MAP   = {1: "A", 2: "B", 3: "C"}

_dss_lock = threading.Lock()       # DSS engine เป็น singleton — ต้องใช้ lock เมื่อมีหลาย thread
_AW_SIZES = sorted(AW_IMP.keys())  # [50, 95] — ขนาดสายที่รองรับใน impedance table

# มิเตอร์ EV charger — GIS ไม่มี field ระบุ แต่ KWP ถูกกำหนดเป็น 7.4 kW (32A x 230V)
# ข้อมูลจริง 1,500 หม้อแปลง: ช่วง 7.0-8.2 kW มี 166 ตัว (7.4 kW ตรงเป๊ะ 160 ตัว)
# ขณะที่ช่วง 5-7 kW มีแค่ 35 ตัว -> ใช้ช่วงนี้แยก EV ออกจากโหลดบ้านทั่วไป
# Step 5 (เพิ่มขนาดสาย) ทำเฉพาะเมื่อมี EV เท่านั้น — โหลดบ้านปกติไม่คุ้มเปลี่ยนสาย
EV_KW_MIN = 7.0
EV_KW_MAX = 8.2


def _is_ev_kw(kw: float) -> bool:
    return EV_KW_MIN <= kw <= EV_KW_MAX


# ── เกณฑ์เชิงมาตรฐานการออกแบบ (Design Check) ────────────────────────────────
# หลักออกแบบ: สายเมน/สายย่อยที่มีความยาวหรือจำนวนผู้ใช้ไฟมากพอ ควรเดินครบเฟสของ
# หม้อแปลง แม้การย้ายเฟสมิเตอร์จะทำให้ %Unbalance ผ่านเกณฑ์ไปแล้วก็ตาม
# (สายไม่ครบเฟส = ขยายโหลดในอนาคตไม่ได้ ต้องย้ายเฟสซ้ำทุกครั้งที่มีผู้ใช้ไฟใหม่)
#
# เทียบกับ "เฟสที่หม้อแปลงมีจริง" (tx_mask) ไม่ใช่ 3 เฟสเสมอ — หม้อแปลง 2 เฟส
# มีสาย 3 เฟสไม่ได้อยู่แล้ว (~44% ของระบบ) ถ้าเทียบกับ 3 เฟสจะติดธงทั้งหมด
#
# ซอยสั้นที่มีบ้านไม่กี่หลังไม่คุ้มเดินเพิ่มเฟส จึงนับเฉพาะกลุ่มที่ใหญ่พอ
DESIGN_MIN_COMP_LEN_M  = 100.0   # ความยาวรวมของกลุ่มสาย (m)
DESIGN_MIN_COMP_METERS = 5       # หรือจำนวนมิเตอร์ในกลุ่ม
PD_TO_NODE  = {4: 1, 2: 2, 1: 3}        # single-phase PHASEDESIGNATION → DSS node#
NODE_TO_PD  = {1: 4, 2: 2, 3: 1}
PD_TO_PHASE = {4: "A", 2: "B", 1: "C"}
SINGLE_PDS  = frozenset({1, 2, 4})       # single-phase PHASEDESIGNATION values

_PD_LABEL = {7: "ABC", 6: "AB", 5: "CA", 4: "A", 3: "BC", 2: "B", 1: "C"}

# สีประจำเฟสสำหรับหมุดมิเตอร์ที่ต้องย้าย (A แดง / B เหลือง / C น้ำเงิน)
_PHASE_MARK_COLOR = {4: "#E03131", 2: "#F0B429", 1: "#2F6FD6"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    converged: bool
    node_phase_v: Dict[int, Dict[int, float]]  # node_id → {phase_node: V}
    min_v: float
    tr_loading_pct: float
    low_v_nodes: List[int]                     # nodes where min_phase_V < threshold
    phase_kw: Dict[int, float] = field(default_factory=dict)  # {1:kWA, 2:kWB, 3:kWC}
    phase_imbalance_pct: float = 0.0
    phase_kva: Dict[int, float] = field(default_factory=dict)          # kVA ต่อเฟส ที่ขั้วทุติยภูมิ
    phase_rating_kva: Dict[int, float] = field(default_factory=dict)   # พิกัดต่อเฟส (kVA)
    phase_loading_pct: Dict[int, float] = field(default_factory=dict)  # % ของพิกัดเฟส
    max_phase_loading_pct: float = 0.0
    error: str = ""


@dataclass
class MeterInfo:
    feat_idx: int
    node_id: int
    peano: str
    current_pd: int    # PHASEDESIGNATION in JSON (1,2,4 for single-phase)
    conductor_pd: int  # phases physically on connecting conductor
    kw: float

    @property
    def is_single_phase(self) -> bool:
        return self.current_pd in SINGLE_PDS

    @property
    def movable_pds(self) -> List[int]:
        """Other single-phase PDs available on the conductor."""
        other = self.conductor_pd & ~self.current_pd
        return [pd for pd in (4, 2, 1) if other & pd]


@dataclass
class PhaseMove:
    meter: MeterInfo
    from_pd: int
    to_pd: int
    delta_min_v: float
    delta_imbalance: float
    level: int = -1      # depth ของชุมสายที่ตัดสินใจย้าย (0 = เมน, มาก = branch ปลาย)
    scope: str = ""      # ป้ายกำกับกลุ่มที่จัดสมดุล (branch / main line)
    stage: str = "greedy"  # "bottomup" = ไล่จาก branch ขึ้นเมน, "greedy" = เก็บงานแรงดันต่ำ


@dataclass
class UpgradeResult:
    edge: Tuple[int, int]                # entry edge เข้าสู่ subtree แรงดันต่ำ (representative)
    feat_idx: int                       # feat ของ entry edge
    from_size: int                      # ขนาดเดิมของ entry edge
    to_size: int                        # ขนาดใหม่ (เท่ากันทุก segment)
    n_affected: int
    result: SimResult
    delta_min_v: float
    upgraded_edges: List[Tuple[int, int, int]] = field(default_factory=list)  # (feat_idx, u, v) ทุก segment ที่อัปเกรด
    from_sizes: Dict[int, int] = field(default_factory=dict)                   # feat_idx → ขนาดเดิม
    ev_peanos: List[str] = field(default_factory=list)                        # มิเตอร์ EV ที่เป็นเหตุให้เพิ่มขนาดสาย


@dataclass
class PhaseAddResult:
    feat_idx: int
    edge: Tuple[int, int]          # target segment (u→v) ที่ low-V อยู่
    from_pd: int
    to_pd: int
    n_affected: int
    meter_moves: List[Tuple[int, int]]          # (feat_idx, new_pd) เฉพาะใน subtree(v)
    upgraded_edges: List[Tuple[int, int, int]]  # (feat_idx, u, v) ทุก edge บน path ที่ต้องอัปเกรด
    result: SimResult
    delta_min_v: float
    delta_imbalance: float = 0.0   # imbal_before − imbal_after (มากกว่า 0 = ดีขึ้น)
    n_comps: int = 1               # จำนวนกลุ่มสายที่ตัวเลือกนี้ครอบคลุม
    design_fixed: int = 0          # จำนวนกลุ่มที่ผิดหลักออกแบบซึ่งตัวเลือกนี้แก้ให้ครบเฟส


@dataclass
class DesignComp:
    """กลุ่มสาย LV ที่เฟสไม่ครบตามหม้อแปลง (1 connected component)"""
    idx: int
    nodes: Set[int]
    edges: List[Tuple[int, int, int, int]]   # (feat_idx, u, v, pd)
    length_m: float
    n_meters: int
    kw: float
    pds: Set[int]                            # PHASEDESIGNATION ที่พบในกลุ่ม
    violates: bool                           # ใหญ่พอที่จะถือว่าผิดหลักออกแบบ

    @property
    def label(self) -> str:
        return (f"กลุ่ม {self.idx+1}: {self.length_m:.0f} m  {len(self.edges)} segment  "
                f"{self.n_meters} มิเตอร์  {self.kw:.1f} kW  "
                f"เฟส {'/'.join(_PD_LABEL.get(p, str(p)) for p in sorted(self.pds))}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _phase_imbalance_pct(phases: Dict[int, float]) -> float:
    if len(phases) < 2:
        return 0.0
    vals = list(phases.values())
    mean_v = sum(vals) / len(vals)
    return max(abs(v - mean_v) for v in vals) / mean_v * 100.0 if mean_v else 0.0


def _conductor_pd_at_node(net: NetworkGraph, raw: dict, node_id: int) -> int:
    """Union of PHASEDESIGNATION bits for all LC lines touching node_id."""
    result = 0
    for feat_idx, (u, v) in net.lc_feat_edges.items():
        if u == node_id or v == node_id:
            pd = int(get_attr(raw["features"][feat_idx], "PHASEDESIGNATION", 7) or 7)
            result |= pd
    return result if result else 7


class BusNodeMapper:
    """
    map ชื่อ bus ของ DSS → node ของ NetworkGraph ด้วยพิกัด (SetBusXY ที่ Runopendss เขียนไว้)

    เดิม map ด้วยลำดับ BFS (ชื่อ S000123) ซึ่งถูกก็ต่อเมื่อ DSS กับ NetworkGraph ใช้ชุดสายเดียวกันเป๊ะ —
    ตอน Runopendss ยังเอาสายไฟสาธารณะ (SUBTYPECODE=2) เข้ามา ลำดับเพี้ยน → แรงดัน map ผิดโหนด 25–100%
    ในหม้อแปลงที่มีสายไฟสาธารณะ (~17% ของทั้งหมด). ใช้พิกัดจึงไม่ขึ้นกับลำดับ BFS (แบบ TransferOptimizer)
    """

    def __init__(self, net: NetworkGraph, tol_m: Optional[float] = None) -> None:
        self.ids = list(net.node_coords.keys())
        self.tree = cKDTree(np.array([net.node_coords[i] for i in self.ids], dtype=float))
        # DSS เขียนพิกัดเป็น centroid ของ cluster เดียวกัน (ทศนิยม 3 ตำแหน่ง) → ต่างกันระดับมม.
        self.tol_m = tol_m if tol_m is not None else max(2.0 * float(net.snap_tol), 1.0)
        self.source_node = net.transformer_node

    def map_buses(self, bus_xy: Dict[str, Optional[Tuple[float, float]]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for bus, xy in bus_xy.items():
            b = bus.lower()
            if b == "source":                    # ขั้วทุติยภูมิหม้อแปลง (DSS ไม่เขียนพิกัดให้)
                out[b] = self.source_node
                continue
            if xy is None:                       # bus มิเตอร์ (m_...) / sourcebus — ไม่ map
                continue
            dist, idx = self.tree.query(xy)
            if dist <= self.tol_m:
                out[b] = self.ids[int(idx)]
        return out


def _build_bus_to_node(net: NetworkGraph) -> "BusNodeMapper":
    """ตัว map bus DSS → node ด้วยพิกัด (ชื่อฟังก์ชันเดิมเพื่อไม่ต้องแก้ call site)"""
    return BusNodeMapper(net)


def _tx_phase_rating_from_raw(raw: dict) -> Dict[int, float]:
    """พิกัดรายเฟสหม้อแปลงจาก JSON (PHASEA_KVA / PHASEB_KVA / PHASEC_KVA) → {DSS node: kVA}
    เฟสที่ค่าเป็น 0/ว่าง จะไม่อยู่ใน dict (ให้ _simulate ใช้ kVA ขดลวดจาก DSS แทน)"""
    for f in raw.get("features", []):
        tag = str(get_attr(f, "TAG", "") or "").upper()
        if "XF" in tag and has_point(f):
            out: Dict[int, float] = {}
            for n, fld in _PHASE_KVA_FIELDS.items():
                try:
                    v = float(get_attr(f, fld, 0.0) or 0.0)
                except (TypeError, ValueError):
                    v = 0.0
                if v > 0:
                    out[n] = v
            return out
    return {}


def _simulate(
    json_path: str,
    bus_to_node: Dict[str, int],
    rated_kva: float,
    min_v_thr: float,
    snap_tol: float = 2.0,
    phase_rating_kva: Optional[Dict[int, float]] = None,
) -> SimResult:
    """Run OpenDSS on json_path; return SimResult.
    bus_to_node — BusNodeMapper (map ด้วยพิกัด) หรือ dict ชื่อ bus → node แบบเดิม
    phase_rating_kva — พิกัดรายเฟสจาก JSON (PHASEx_KVA); ไม่ส่งมา = ใช้ kVA ขดลวดจาก DSS"""
    tmp = tempfile.mkdtemp(prefix="popt_")
    try:
        dss_path = str(Path(tmp) / "sim.dss")
        with _dss_lock:
            with contextlib.redirect_stdout(io.StringIO()):
                dss_file, _ = convert_json_to_dss_ordered(json_path, dss_path, snap_tol=snap_tol)
                solve_with_opendss(dss_file)

            if not odss.Solution.Converged():
                return SimResult(converged=False, node_phase_v={}, min_v=0.0,
                                 tr_loading_pct=0.0, low_v_nodes=[], error="not converged")

            dss_vmap = _collect_dss_bus_phase_voltages()
            # พิกัดของแต่ละ bus (SetBusXY) — ใช้ map → node แบบไม่พึ่งลำดับ BFS
            bus_xy: Dict[str, Optional[Tuple[float, float]]] = {}
            for _b in dss_vmap:
                odss.Circuit.SetActiveBus(_b)
                bus_xy[_b] = ((odss.Bus.X(), odss.Bus.Y())
                              if odss.Bus.Coorddefined() else None)

            # Transformer per-phase kW (secondary terminal)
            # หม้อแปลง 1/2 เฟส จะถูกแยกเป็นหลาย Transformer object (คนละเฟส) —
            # ต้องวนรวมทุกก้อน และใช้ NodeOrder จริงแทนการอ้าง index ตรง ๆ (ไม่งั้น
            # ก้อนเฟสเดียวจะ mislabel เป็นเฟส A เสมอ และก้อนอื่นจะถูกมองข้ามไปเลย)
            phase_kw: Dict[int, float] = {}
            phase_kva: Dict[int, float] = {}
            phase_rating: Dict[int, float] = {}
            phase_imbal = 0.0
            try:
                tx_names = odss.Transformers.AllNames()
                for _tx_name in tx_names:
                    odss.Circuit.SetActiveElement(f"Transformer.{_tx_name}")
                    pq_flat    = odss.CktElement.Powers()
                    pq         = list(zip(pq_flat[0::2], pq_flat[1::2]))
                    nodeorder  = odss.CktElement.NodeOrder()
                    ncond      = odss.CktElement.NumConductors()
                    nterm      = odss.CktElement.NumTerminals()
                    base       = (nterm - 1) * ncond  # secondary terminal
                    # พิกัดต่อเฟสสำรอง = kVA ขดลวดทุติยภูมิ / จำนวนเฟส (ใช้เมื่อ JSON ไม่มี PHASEx_KVA)
                    odss.Transformers.Name(_tx_name)
                    odss.Transformers.Wdg(nterm)
                    unit_kva = odss.Transformers.kVA()
                    nph      = max(odss.CktElement.NumPhases(), 1)
                    for k in range(ncond):
                        idx = base + k
                        if idx >= len(nodeorder) or idx >= len(pq):
                            continue
                        node = int(nodeorder[idx])
                        if node == 0:
                            continue
                        phase_kw[node] = phase_kw.get(node, 0.0) + abs(pq[idx][0])
                        phase_kva[node] = (phase_kva.get(node, 0.0)
                                           + math.hypot(pq[idx][0], pq[idx][1]))
                        phase_rating[node] = phase_rating.get(node, 0.0) + unit_kva / nph
                if len(phase_kw) >= 2:
                    vals = list(phase_kw.values())
                    mv   = sum(vals) / len(vals)
                    phase_imbal = (max(abs(v - mv) for v in vals) / mv * 100.0
                                   if mv else 0.0)
            except Exception:
                pass

            tr_loading = query_transformer_loading(rated_kva)
            # พิกัดรายเฟส: ใช้ PHASEx_KVA จาก JSON เป็นหลัก, เฟสที่ไม่มีค่าใช้ของ DSS
            if phase_rating_kva:
                for _n, _rv in phase_rating_kva.items():
                    if _rv > 0 and _n in phase_rating:
                        phase_rating[_n] = _rv
            phase_loading = {n: 100.0 * phase_kva.get(n, 0.0) / r
                             for n, r in phase_rating.items() if r > 0}

        # Map bus names → node IDs outside lock (pure Python, no DSS calls)
        if isinstance(bus_to_node, BusNodeMapper):
            b2n = bus_to_node.map_buses(bus_xy)
        else:                                   # dict ชื่อ bus → node แบบเดิม (สคริปต์เก่า)
            b2n = {str(b).lower(): n for b, n in bus_to_node.items()}
        node_phase_v: Dict[int, Dict[int, float]] = {}
        for bus, phases in dss_vmap.items():
            nid = b2n.get(bus.lower())
            if nid is None:
                continue
            old = node_phase_v.get(nid)
            # หลาย bus ตกโหนดเดียวกัน → เก็บตัวที่แรงดันต่ำสุด (มองแง่ร้าย)
            if old is None or (phases and (not old or min(phases.values()) < min(old.values()))):
                node_phase_v[nid] = phases

        min_v_global = min(
            (min(ph.values()) for ph in node_phase_v.values() if ph),
            default=0.0,
        )
        low_v_nodes = [
            nid for nid, ph in node_phase_v.items()
            if ph and min(ph.values()) < min_v_thr
        ]

        return SimResult(
            converged=True,
            node_phase_v=node_phase_v,
            min_v=min_v_global,
            tr_loading_pct=tr_loading,
            low_v_nodes=low_v_nodes,
            phase_kw=phase_kw,
            phase_imbalance_pct=phase_imbal,
            phase_kva=phase_kva,
            phase_rating_kva=phase_rating,
            phase_loading_pct=phase_loading,
            max_phase_loading_pct=max(phase_loading.values(), default=0.0),
        )
    except Exception as exc:
        return SimResult(converged=False, node_phase_v={}, min_v=0.0,
                         tr_loading_pct=0.0, low_v_nodes=[], error=str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _sim_modified_json(
    modified_raw: dict,
    bus_to_node: Dict[str, int],
    rated_kva: float,
    min_v_thr: float,
    snap_tol: float,
) -> SimResult:
    """Write modified_raw to temp file and simulate."""
    tmp_json = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump(modified_raw, tmp_json, ensure_ascii=False)
    tmp_json.close()
    try:
        return _simulate(tmp_json.name, bus_to_node, rated_kva, min_v_thr, snap_tol,
                         phase_rating_kva=_tx_phase_rating_from_raw(modified_raw))
    finally:
        os.unlink(tmp_json.name)


_LETTER_TO_PD = {"A": 4, "B": 2, "C": 1}


def _meter_current_pd(feat: dict) -> int:
    """
    Determine effective single-phase PHASEDESIGNATION for a meter feature.
    DSS generator uses PHASE attribute (A/B/C) before PHASEDESIGNATION,
    so we mirror the same priority to get the correct current phase.
    """
    letter = str(get_attr(feat, "PHASE", "") or "").strip().upper()
    if letter in _LETTER_TO_PD:
        return _LETTER_TO_PD[letter]
    pd = int(get_attr(feat, "PHASEDESIGNATION", 7) or 7)
    # If not a clean single-phase PD, infer from first available phase bit
    if pd not in SINGLE_PDS:
        for bit in (4, 2, 1):
            if pd & bit:
                return bit
    return pd


def _apply_meter_phase(raw_features: list, feat_idx: int, new_pd: int) -> None:
    """Update both PHASEDESIGNATION and PHASE on a meter feature in-place."""
    attrs = raw_features[feat_idx].setdefault("attributes", {})
    attrs["PHASEDESIGNATION"] = new_pd
    attrs["PHASE"] = PD_TO_PHASE.get(new_pd, "A")


def _build_meter_inventory(net: NetworkGraph, raw: dict) -> List[MeterInfo]:
    """Collect all meters with their phase and conductor availability."""
    meters: List[MeterInfo] = []
    for feat_idx, node_id in net.load_feat_nodes.items():
        feat = raw["features"][feat_idx]
        peano = str(get_attr(feat, "PEANO", "") or "").strip()
        pd = _meter_current_pd(feat)
        kw = float(get_attr(feat, "KWP", 0.0) or 0.0)
        cond_pd = _conductor_pd_at_node(net, raw, node_id)
        # มิเตอร์ที่มีสายบริการหลายเส้น (ต่อหลายจุด) ใช้ได้เฉพาะเฟสที่มีครบทุกจุดต่อ
        for _cn in getattr(net, "load_feat_conn_nodes", {}).get(feat_idx, [])[1:]:
            _both = cond_pd & _conductor_pd_at_node(net, raw, _cn)
            if _both:
                cond_pd = _both
        meters.append(MeterInfo(
            feat_idx=feat_idx, node_id=node_id, peano=peano,
            current_pd=pd, conductor_pd=cond_pd, kw=kw,
        ))
    return meters


def _path_to_node(net: NetworkGraph, target: int) -> List[Tuple[int, int, int]]:
    """Return [(feat_idx, u, v), ...] for edges on BFS path from transformer_node to target.
    u→v follows BFS tree direction (not necessarily the physical cable direction)."""
    try:
        nodes = nx.shortest_path(net.G, source=net.transformer_node, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    # Bidirectional lookup — lc_feat_edges may store physical cable direction (reversed vs BFS)
    edge_to_feat: Dict[Tuple[int,int], int] = {}
    for fi, (u, v) in net.lc_feat_edges.items():
        edge_to_feat[(u, v)] = fi
        edge_to_feat[(v, u)] = fi
    return [(edge_to_feat[(nodes[i], nodes[i+1])], nodes[i], nodes[i+1])
            for i in range(len(nodes) - 1)
            if (nodes[i], nodes[i+1]) in edge_to_feat]


def _overload_excess(r: SimResult, limit: float) -> float:
    """ส่วนที่เฟสหนักสุดเกินพิกัด (percentage point) — 0 ถ้าไม่เกิน"""
    return max(0.0, r.max_phase_loading_pct - limit)


def _overload_ok(before: SimResult, after: SimResult,
                 limit: float = DEFAULT_MAX_PHASE_LOADING_PCT) -> bool:
    """
    พิกัดรายเฟสหม้อแปลง: การแก้ไขใด ๆ ห้ามทำให้เฟสไหนเกินพิกัด
    ถ้าเกินพิกัดอยู่แล้วตั้งแต่ก่อนแก้ ต้องไม่แย่ลง (เผื่อ 0.5%)
    """
    if not after.converged:
        return False
    if after.max_phase_loading_pct <= limit:
        return True
    return after.max_phase_loading_pct <= before.max_phase_loading_pct + 0.5


def _tr_rating_advice(r: Optional[SimResult], limit: float) -> List[str]:
    """คำแนะนำด้านพิกัดหม้อแปลงจากผลสุดท้าย — เคสที่ขั้นตอนฝั่ง LV แก้ไม่ได้"""
    if r is None or not r.converged or not r.phase_loading_pct:
        return []
    over = {n: v for n, v in r.phase_loading_pct.items() if v > limit}
    if not over:
        return []
    tot_rating = sum(r.phase_rating_kva.values())
    T = 100.0 * sum(r.phase_kva.values()) / tot_rating if tot_rating > 0 else 0.0
    ph = ", ".join(f"เฟส {PHASE_MAP.get(n, '?')} {v:.0f}%" for n, v in sorted(over.items()))
    msgs = [f"หม้อแปลงเกินพิกัดรายเฟส: {ph} (เกณฑ์ {limit:.0f}%)"]
    if T > limit:
        msgs.append(f"โหลดรวม {T:.0f}% ของพิกัด — ย้ายเฟสมิเตอร์/เพิ่มเฟสสาย LV แก้ไม่ได้ "
                    "ต้องเพิ่มขนาดหม้อแปลง")
    elif len(r.phase_rating_kva) < 3:
        msgs.append(f"หม้อแปลง {len(r.phase_rating_kva)} เฟส โหลดรวม {T:.0f}% — เกลี่ยโหลดได้ไม่พอ "
                    "พิจารณาเพิ่มเฟสหม้อแปลง หรือเพิ่มขนาดหม้อแปลง")
    else:
        msgs.append(f"โหลดรวม {T:.0f}% แต่เกลี่ยเฟสได้ไม่พอ (สายไม่มีเฟสให้ย้าย) — "
                    "พิจารณาเพิ่มเฟสสาย หรือเพิ่มขนาดหม้อแปลง")
    return msgs


def _score_result(before: SimResult, after: SimResult,
                  low_v_mode: bool = True, min_v_thr: float = 0.0,
                  max_phase_loading_pct: float = DEFAULT_MAX_PHASE_LOADING_PCT) -> float:
    """
    Score improvement (higher = better).

    low_v_mode=True  — มีโหนดแรงดันต่ำ → Vmin สำคัญกว่า imbalance 10 เท่า (เหมือนเดิม)
    low_v_mode=False — ไม่มีโหนดแรงดันต่ำ → เป้าหมายคือ imbalance ล้วน
                       Vmin ไม่ถูกให้คะแนน (ΔV ระดับ 0.05V คือ noise เชิงตัวเลข
                       ไม่ใช่การปรับปรุงจริง) แต่ยังลงโทษถ้า Vmin จะร่วงต่ำกว่าเกณฑ์
    """
    if not after.converged:
        return -999.0
    # ห้ามทำให้เฟสหม้อแปลงเกินพิกัด / ให้คะแนนส่วนที่ลดการเกินพิกัดได้ (นับเท่า imbalance)
    if not _overload_ok(before, after, max_phase_loading_pct):
        return -999.0
    d_ov  = (_overload_excess(before, max_phase_loading_pct) -
             _overload_excess(after, max_phase_loading_pct))
    d_imb = before.phase_imbalance_pct - after.phase_imbalance_pct + d_ov
    if low_v_mode:
        return (after.min_v - before.min_v) * 10.0 + d_imb
    penalty = 0.0
    if after.min_v < min_v_thr:                     # ห้ามสร้างปัญหาแรงดันใหม่
        penalty = (min_v_thr - after.min_v) * 10.0
    return d_imb - penalty


# ─────────────────────────────────────────────────────────────────────────────
# Step 1a — Bottom-up Phase Balance  (ไล่จากไลน์ branch ขึ้นมาถึงไลน์เมน)
# ─────────────────────────────────────────────────────────────────────────────
#
# แนวคิด — เหมือนที่ช่างเดินตรวจจริง:
#   1. เริ่มจากปลายกิ่ง (branch) ที่ลึกที่สุด รวบมิเตอร์ทั้งกิ่งมาเป็น "กอง" เดียว
#      ที่ชุมสาย (junction) ของกิ่งนั้น แล้วจัดสมดุลภายในกิ่งก่อน
#   2. ไต่ขึ้นมาทีละชั้น — ชั้นบนเห็นโหลดของกิ่งล่างที่จัดเสร็จแล้วเป็นค่าคงที่
#      แล้วใช้มิเตอร์ที่อยู่บนช่วงสายของตัวเองชดเชยส่วนที่กิ่งล่างเอียง
#   3. จบที่ไลน์เมน (root) ซึ่งเห็นภาพรวมทั้งหม้อแปลง
#
# การเลือกมิเตอร์ในแต่ละกอง: ไล่จากตัวที่โหลด (KWP) มากสุดก่อน และย้ายก็ต่อเมื่อ
# ย้ายแล้วทำให้เฟสในกองนั้นสมดุลขึ้นจริง → ได้จำนวนครั้งการย้ายน้อยที่สุด
#
# คิดด้วยเลข kW ล้วน ไม่ต้องยิง OpenDSS ต่อ 1 การย้าย (ของเดิมยิง 1 sim/มิเตอร์)
# แล้วค่อย verify ด้วย sim เดียวตอนจบ

_PD_ORDER = (4, 2, 1)   # A, B, C

# ย้ายมิเตอร์ 1 ตัว = ส่งชุดปฏิบัติงานออกหน้างาน 1 ครั้ง
# ถ้าย้ายแล้วลดความเบี่ยงเบนต่อเฟส (RMS) ได้ไม่ถึงค่านี้ ถือว่าไม่คุ้ม — ไม่ย้าย
MIN_MOVE_GAIN_KW = 0.02

# ตัวเกลี่ยคิดจาก KWP ของมิเตอร์ แต่ผล OpenDSS (รวม loss/แรงดัน) มักสูงกว่าราว 0.5%
# จึงตั้งเป้าภายในต่ำกว่าเกณฑ์ 1% เพื่อให้ผล verify ต่ำกว่าเกณฑ์จริง
# ไม่ต้องให้ greedy (ช้า) มาเก็บเศษ
BALANCE_MARGIN_PCT = 1.0

# เป้าของแต่ละกองตอนกระจายเฟส (Phase Addition) = ผสมระหว่าง
#   "แบ่งเท่ากันทุกเฟส" (สมดุลในกิ่ง เฟสที่เพิ่งเดินเพิ่มมีโหลดจริง)  กับ
#   "สัดส่วนที่ทำให้หม้อแปลงสมดุล" (ชดเชยโหลดนอกกลุ่มที่เอียงอยู่)
# 0 = เอาแต่สมดุลในกิ่ง (หม้อแปลงอาจเอียง), 1 = เอาแต่หม้อแปลง (กิ่งอาจไม่มีโหลดบางเฟส)
_POOL_SHARE_W = 0.75

# กองที่มีมิเตอร์ตั้งแต่จำนวนนี้ ให้จัดสมดุลตรงนั้นเลย ไม่ต้องยกขึ้นไปรวมที่ชุมสาย
# (ลูกซอยกลางทางที่มีผู้ใช้ไฟมากพอ ควรมีโหลดครบทุกเฟสในตัวเอง)
# ตั้งสูงมาก = ปิดการทำงาน (กลับไปรวมที่ชุมสายอย่างเดียว)
_POOL_FLUSH_N = 8


def _tx_available_pds(baseline: SimResult) -> List[int]:
    """
    เฟสที่หม้อแปลง "มีจริง" — อ่านจาก node ของ phase_kw ที่ DSS solve ออกมา
    หม้อแปลง 1φ/2φ จะมีไม่ครบ 3 เฟส ห้ามย้ายมิเตอร์ไปเฟสที่ไม่มีไฟ
    (ไม่งั้นโหลดจะกิน 0 kW แล้วดู "สมดุล" แบบหลอก ๆ)
    """
    pds = [NODE_TO_PD[n] for n in sorted(baseline.phase_kw) if n in NODE_TO_PD]
    return pds or list(_PD_ORDER)


def _spread_kw(acc: Dict[int, float]) -> float:
    """ค่าเบี่ยงเบนสูงสุดจากค่าเฉลี่ย (kW) — นิยามเดียวกับ phase_imbalance_pct"""
    if len(acc) < 2:
        return 0.0
    vals = list(acc.values())
    mean_v = sum(vals) / len(vals)
    return max(abs(v - mean_v) for v in vals)


def _spread_pct(acc: Dict[int, float]) -> float:
    if len(acc) < 2:
        return 0.0
    vals = list(acc.values())
    mean_v = sum(vals) / len(vals)
    return (_spread_kw(acc) / mean_v * 100.0) if mean_v > 0 else 0.0


def _rms_kw(acc: Dict[int, float]) -> float:
    """
    ค่าเบี่ยงเบนกำลังสองเฉลี่ย (kW) — ใช้เป็น "เป้าหมายตอนค้นหา" แทน max-deviation

    max-deviation มีที่ราบ (plateau): A/B/C = 4/2/0 กับ 3/3/0 ได้ค่าเท่ากัน (=2)
    ทำให้ descent หยุดที่ 4/2/0 ทั้งที่ 3/3/0 กระแสนิวทรัลต่ำกว่าชัดเจน
    RMS เป็นฟังก์ชันนูนเคร่ง ไม่มีที่ราบ จุดต่ำสุดตรงกับ max-deviation
    และมีหน่วยเป็น kW จึงตั้งเกณฑ์ "ย้ายแล้วคุ้มไหม" ได้ตรง ๆ
    (ยังรายงานผลด้วย _spread_pct ตามนิยามเดิมของ phase_imbalance_pct)
    """
    if len(acc) < 2:
        return 0.0
    vals = list(acc.values())
    mean_v = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals))


def _meter_targets(m: MeterInfo, tx_pds: List[int]) -> List[int]:
    """เฟสที่มิเตอร์ตัวนี้ย้ายไปได้จริง = สายที่ต่ออยู่รองรับ ∩ หม้อแปลงมี"""
    return [pd for pd in tx_pds if m.conductor_pd & pd]


def _descend_balance(
    pool: List[MeterInfo],
    acc: Dict[int, float],
    assign: Dict[int, int],
    tx_pds: List[int],
    max_passes: int = 8,
    min_gain_kw: float = MIN_MOVE_GAIN_KW,
    target_pct: float = 0.0,
    glob: Optional[Dict[int, float]] = None,
) -> int:
    """
    จัดสมดุลกอง `pool` โดยย้ายมิเตอร์ทีละตัว — ไล่ตัวโหลดเยอะสุดก่อน
    ย้ายก็ต่อเมื่อลด RMS ของ acc ได้ ≥ min_gain_kw (ไม่ไล่เก็บเศษที่ไม่คุ้มค่าแรง)

    `acc` ต้องรวม kW ของทุกตัวใน pool ที่เฟสตาม `assign` ไว้แล้วก่อนเรียก
    คืนจำนวนครั้งที่ย้าย (นับซ้ำได้ถ้าตัวเดิมถูกย้ายหลายรอบ)
    """
    ordered = sorted(pool, key=lambda m: -m.kw)     # โหลดเยอะสุดก่อน
    n_applied = 0
    # target_pct > 0 → หยุดทันทีเมื่อ %Unbalance ต่ำกว่าเป้า ทั้งของกองนี้ (acc)
    # หรือของทั้งหม้อแปลง (glob) — ไม่ไล่ต่อจนเป็น 0%
    def _reached() -> bool:
        if target_pct <= 0:
            return False
        return (_spread_pct(acc) < target_pct or
                (glob is not None and _spread_pct(glob) < target_pct))

    if _reached():
        return 0
    for _ in range(max_passes):
        changed = False
        for m in ordered:
            if m.kw <= 0.0:
                continue
            opts = _meter_targets(m, tx_pds)
            if len(opts) < 2:
                continue                            # ไม่มีทางเลือก — สายไม่มีเฟสอื่น
            frm  = assign[m.feat_idx]
            base = _rms_kw(acc)
            best_to, best_gain = frm, min_gain_kw
            for to in opts:
                if to == frm:
                    continue
                acc[frm] -= m.kw
                acc[to]  += m.kw
                gain = base - _rms_kw(acc)
                acc[to]  -= m.kw
                acc[frm] += m.kw
                if gain > best_gain:
                    best_gain, best_to = gain, to
            if best_to != frm:
                acc[frm]     -= m.kw
                acc[best_to] += m.kw
                if glob is not None:
                    glob[frm]     -= m.kw
                    glob[best_to] += m.kw
                assign[m.feat_idx] = best_to
                n_applied += 1
                changed = True
                if _reached():
                    return n_applied
        if not changed:
            break
    return n_applied


def _bottom_up_pass(
    net: NetworkGraph,
    raw: dict,
    baseline: SimResult,
    bus_to_node: Dict[str, int],
    min_v_thr: float,
    snap_tol: float = 2.0,
    max_imbalance_pct: float = DEFAULT_MAX_IMBALANCE_PCT,
    max_passes: int = 8,
    widen: bool = True,
    min_gain_kw: float = MIN_MOVE_GAIN_KW,
    target_pct: float = 0.0,
    max_phase_loading_pct: float = DEFAULT_MAX_PHASE_LOADING_PCT,
) -> Tuple[dict, List[PhaseMove], Optional[SimResult]]:
    """
    ไล่สมดุลจากไลน์ branch ขึ้นมาถึงไลน์เมน 1 รอบ
    target_pct > 0 → แต่ละกองหยุดเมื่อ %Unbalance ต่ำกว่า target_pct
    target_pct = 0 → เกลี่ยเต็มที่

    Returns (modified_raw, moves, verify_result)
    moves ว่าง = ไม่ย้าย/ผลไม่ผ่านเกณฑ์ (คืน raw เดิม)
    """
    tx_pds = _tx_available_pds(baseline)
    if len(tx_pds) < 2:
        print("  [Bottom-up] หม้อแปลงมีเฟสเดียว — ไม่มีอะไรให้สมดุล")
        return raw, [], None

    current_raw = copy.deepcopy(raw)
    meters = _build_meter_inventory(net, current_raw)
    G, root = net.G, net.transformer_node

    at_node: Dict[int, List[MeterInfo]] = defaultdict(list)
    for m in meters:
        at_node[m.node_id].append(m)

    # เฟสตั้งต้น = เฟสปัจจุบัน (ถ้าอยู่เฟสที่หม้อแปลงไม่มี ให้ดึงเข้าเฟสที่มี)
    assign: Dict[int, int] = {}
    for m in meters:
        opts = _meter_targets(m, tx_pds)
        if m.is_single_phase and m.current_pd in tx_pds:
            assign[m.feat_idx] = m.current_pd
        else:
            assign[m.feat_idx] = opts[0] if opts else tx_pds[0]

    # BFS order + depth จากหม้อแปลง (net.G เป็น tree อยู่แล้ว)
    depth = {root: 0}
    order = [root]
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in G.successors(u):
            depth[v] = depth[u] + 1
            order.append(v)
            dq.append(v)

    in_tree = set(order)
    orphans = [m for m in meters if m.node_id not in in_tree]   # เกาะที่หลุดจาก tree

    def _add_fixed(acc: Dict[int, float], m: MeterInfo) -> None:
        """บวก kW ของมิเตอร์ที่ย้ายไม่ได้ลง acc ตามเฟสที่มันอยู่"""
        if m.is_single_phase:
            acc[assign[m.feat_idx]] += m.kw
        else:                                   # โหลด 3 เฟส — กระจายเท่ากัน
            share = m.kw / len(acc)
            for p in acc:
                acc[p] += share

    def _movable(m: MeterInfo) -> bool:
        return m.is_single_phase and len(_meter_targets(m, tx_pds)) >= 2

    # โหลดรวมทั้งหม้อแปลง ณ เฟสปัจจุบัน — ใช้เช็ก "ต่ำกว่าเกณฑ์แล้วหยุด" ที่ระดับหม้อแปลง
    glob: Dict[int, float] = {pd: 0.0 for pd in tx_pds}
    for m in meters:
        if m.is_single_phase:
            glob[assign[m.feat_idx]] += m.kw
        else:
            for p in glob:
                glob[p] += m.kw / len(glob)

    def _tx_reached() -> bool:
        return target_pct > 0 and _spread_pct(glob) < target_pct

    sub_acc:  Dict[int, Dict[int, float]] = {}
    sub_pend: Dict[int, List[MeterInfo]]  = {}
    sub_mov:  Dict[int, List[MeterInfo]]  = {}
    levels: List[Tuple[int, int, int, float, float, int]] = []
    raw_moves: List[Tuple[MeterInfo, int, int, int, str, float]] = []

    # ไล่ย้อน BFS order = ลูกเสร็จก่อนพ่อเสมอ → branch ลึกสุดถูกจัดก่อน, root ท้ายสุด
    for v in reversed(order):
        acc:  Dict[int, float] = {pd: 0.0 for pd in tx_pds}
        pend: List[MeterInfo]  = []     # มิเตอร์ที่ยังรอจัด (ยกขึ้นไปจัดที่ชุมสาย)
        mov:  List[MeterInfo]  = []     # มิเตอร์ที่ย้ายได้ทั้งหมดใน subtree นี้

        for m in at_node.get(v, []):
            if _movable(m):
                pend.append(m)
                mov.append(m)
            else:
                _add_fixed(acc, m)

        for c in G.successors(v):
            for p in acc:
                acc[p] += sub_acc[c][p]
            pend.extend(sub_pend[c])
            mov.extend(sub_mov[c])

        if v == root:
            for m in orphans:
                if _movable(m):
                    pend.append(m)
                    mov.append(m)
                else:
                    _add_fixed(acc, m)

        # flush เฉพาะที่ชุมสาย (แยกสาย ≥2 ทาง) และที่ไลน์เมน
        # โหนดกลางสาย (out_degree ≤ 1) ไม่ flush — ยกมิเตอร์ขึ้นไปรวมกองที่ชุมสาย
        # เพื่อให้ทั้ง lateral ถูกจัดพร้อมกัน แทนที่จะจัดทีละโหนดซึ่งไม่มีอะไรให้เกลี่ย
        if (G.out_degree(v) >= 2 or v == root) and pend:
            for m in pend:
                acc[assign[m.feat_idx]] += m.kw     # ลงเฟสเดิมก่อน แล้วค่อยเกลี่ย
            before = _spread_pct(acc)
            snap   = dict(assign)

            # หม้อแปลงต่ำกว่าเกณฑ์แล้ว → ไม่ย้ายต่อ (แต่ยังรวม kW ขึ้นไปให้ชั้นบนตามปกติ)
            if not _tx_reached():
                _descend_balance(pend, acc, assign, tx_pds, max_passes,
                                 min_gain_kw, target_pct, glob)

                # ถ้ากองของชั้นนี้ยังเกลี่ยไม่พอ (เช่นกิ่งล่างเอียงมาแล้วแก้ไม่ได้)
                # ให้ยืมมิเตอร์ใน subtree ที่จัดไปแล้วมาช่วยชดเชย
                widen_at = target_pct if target_pct > 0 else max_imbalance_pct
                if (widen and not _tx_reached() and _spread_pct(acc) >= widen_at
                        and len(mov) > len(pend)):
                    _descend_balance(mov, acc, assign, tx_pds, max_passes,
                                     min_gain_kw, target_pct, glob)

            after   = _spread_pct(acc)
            changed = [m for m in mov if assign[m.feat_idx] != snap[m.feat_idx]]
            if changed:
                is_main = (v == root)
                label = ("ไลน์เมน" if is_main
                         else f"branch @node {v} (ลึก {depth[v]})")
                levels.append((v, depth[v], len(changed), before, after, len(pend)))
                for m in changed:
                    raw_moves.append((m, snap[m.feat_idx], assign[m.feat_idx],
                                      depth[v], label, before - after))
            pend = []

        sub_acc[v]  = acc
        sub_mov[v]  = mov
        sub_pend[v] = pend

    # ── รวมการย้ายซ้ำของมิเตอร์ตัวเดียวกันให้เหลือ "เฟสเดิม → เฟสสุดท้าย" ──
    # (ตัวหนึ่งอาจถูกแตะทั้งตอนจัด branch และตอนยืมมาชดเชยที่เมน แต่หน้างานคือย้ายครั้งเดียว)
    moves: List[PhaseMove] = []
    seen: Set[int] = set()
    for m, _f, _t, lvl, label, gain in raw_moves:
        if m.feat_idx in seen:
            continue
        seen.add(m.feat_idx)
        if assign[m.feat_idx] == m.current_pd:
            continue                            # ย้ายไปแล้วย้ายกลับ — ไม่ต้องทำอะไร
        moves.append(PhaseMove(
            meter=m, from_pd=m.current_pd, to_pd=assign[m.feat_idx],
            delta_min_v=0.0, delta_imbalance=gain,
            level=lvl, scope=label, stage="bottomup",
        ))

    est = sub_acc.get(root, {})
    goal = (f"หยุดเมื่อ %Unbalance <{target_pct:.0f}%" if target_pct > 0 else "เกลี่ยเต็มที่")
    print(f"\n[Bottom-up Balance] ไล่จาก branch ขึ้นเมน ({goal}) — "
          f"{len(levels)} กลุ่มที่ปรับ, ย้าย {len(moves)} มิเตอร์")
    for v, d, n, bef, aft, npool in sorted(levels, key=lambda x: -x[1]):
        tag = "ไลน์เมน " if v == root else f"branch@{v}"
        print(f"    {tag:>12}  ลึก {d:>2}  กอง {npool:>3} ตัว  "
              f"ย้าย {n:>3}  imbal {bef:5.1f}% → {aft:5.1f}%")
    if est:
        print("    คาดการณ์ที่หม้อแปลง (kW ตาม KWP): " +
              "  ".join(f"{PD_TO_PHASE[p]}={est[p]:.2f}" for p in _PD_ORDER if p in est) +
              f"   spread={_spread_pct(est):.1f}%")

    if not moves:
        print("    ไม่มีการย้ายที่ทำให้สมดุลขึ้น — ข้าม")
        return raw, [], None

    for mv in moves:
        _apply_meter_phase(current_raw["features"], mv.meter.feat_idx, mv.to_pd)

    result = _sim_modified_json(current_raw, bus_to_node,
                                net.rated_kva, min_v_thr, snap_tol)
    if not result.converged:
        print(f"    [NG] verify ไม่ converge ({result.error}) — ย้อนกลับ")
        return raw, [], result

    d_imb = baseline.phase_imbalance_pct - result.phase_imbalance_pct
    d_v   = result.min_v - baseline.min_v
    print(f"    verify: Vmin={result.min_v:.1f}V ({d_v:+.1f})  "
          f"Imbal={result.phase_imbalance_pct:.1f}% ({d_imb:+.1f})  "
          f"PhMax={result.max_phase_loading_pct:.0f}%")

    # เกณฑ์รับผล — ถ้าเดิมไม่มีปัญหาแรงดัน ห้ามสร้างปัญหาแรงดันใหม่
    if baseline.low_v_nodes:
        ok = d_v > 0 or (d_imb > 0.5 and d_v >= -1.0)
    else:
        ok = d_imb > 0.5 and result.min_v >= min_v_thr
    # พิกัดรายเฟสหม้อแปลง: ห้ามทำให้เฟสใดเกินพิกัด (หรือแย่ลงถ้าเกินอยู่แล้ว)
    # ถ้าเดิมเกินพิกัด ยอมรับผลที่ลดการเกินพิกัดได้ ≥1% แม้ imbalance ลดไม่ถึง 0.5%
    if (not ok and _overload_excess(baseline, max_phase_loading_pct) -
            _overload_excess(result, max_phase_loading_pct) >= 1.0 and
            result.min_v >= min(baseline.min_v, min_v_thr) - 1.0):
        ok = True
    ok = ok and _overload_ok(baseline, result, max_phase_loading_pct)
    if not ok:
        print("    ผลไม่ดีขึ้นตามเกณฑ์ — ย้อนกลับ")
        return raw, [], result

    # sync current_pd ให้ขั้นถัดไปเห็นเฟสใหม่
    for mv in moves:
        mv.meter.current_pd = mv.to_pd
    return current_raw, moves, result


def optimize_phase_balance_bottom_up(
    net: NetworkGraph,
    raw: dict,
    baseline: SimResult,
    bus_to_node: Dict[str, int],
    min_v_thr: float,
    snap_tol: float = 2.0,
    max_imbalance_pct: float = DEFAULT_MAX_IMBALANCE_PCT,
    max_passes: int = 8,
    widen: bool = True,
    min_gain_kw: float = MIN_MOVE_GAIN_KW,
    max_phase_loading_pct: float = DEFAULT_MAX_PHASE_LOADING_PCT,
) -> Tuple[dict, List[PhaseMove], Optional[SimResult]]:
    """
    จัดสมดุลเฟสโดยไล่จากไลน์ branch ขึ้นมาถึงไลน์เมน

      • %Unbalance ต่ำกว่า max_imbalance_pct และแรงดันผ่าน → ไม่ย้ายเลย
      • ไม่งั้นไล่เกลี่ยจาก branch ขึ้นเมน แล้ว "หยุด" ทันทีที่ %Unbalance ของ
        หม้อแปลงต่ำกว่าเกณฑ์ (เผื่อ BALANCE_MARGIN_PCT) — ไม่ไล่ต่อจนเป็น 0%
        ประหยัดจำนวนมิเตอร์ที่ต้องย้าย
      • ถ้าผลที่ได้ยังมีโหนดแรงดันต่ำ → ลองเกลี่ยเต็มที่อีกรอบ
        แล้วใช้รอบที่แก้แรงดันได้ดีกว่า

    Returns (modified_raw, moves, verify_result)
    """
    has_low_v  = bool(baseline.low_v_nodes)
    overloaded = baseline.max_phase_loading_pct > max_phase_loading_pct
    if not has_low_v and not overloaded and baseline.phase_imbalance_pct < max_imbalance_pct:
        print(f"\n[Bottom-up Balance] %Unbalance {baseline.phase_imbalance_pct:.1f}% "
              f"< {max_imbalance_pct:.0f}% และแรงดันผ่าน — ไม่ต้องย้าย")
        return raw, [], None

    kw = dict(snap_tol=snap_tol, max_imbalance_pct=max_imbalance_pct,
              max_phase_loading_pct=max_phase_loading_pct,
              max_passes=max_passes, widen=widen, min_gain_kw=min_gain_kw)
    target = max(0.0, max_imbalance_pct - BALANCE_MARGIN_PCT)
    # พิกัดรายเฟส: ถ้าหยุดที่ %Unbalance = target เฟสหนักสุดจะโหลดราว T x (1 + target)
    # (T = kVA รวม / พิกัดรวม) → ถ้าจะเกินพิกัดเฟส ต้องเกลี่ยให้ต่ำกว่านั้น
    tot_rating = sum(baseline.phase_rating_kva.values())
    T = 100.0 * sum(baseline.phase_kva.values()) / tot_rating if tot_rating > 0 else 0.0
    if T >= max_phase_loading_pct:
        # โหลดรวมเกินพิกัดหม้อแปลงแล้ว — เกลี่ยเฟสยังไงก็ยังเกิน ต้องเพิ่มขนาดหม้อแปลง
        # จึงไม่เกลี่ยเต็มที่ (ย้ายมิเตอร์เปล่า ๆ) ใช้เป้า %Unbalance ปกติ
        print()
        print(f"[Bottom-up Balance] โหลดรวม {T:.0f}% เกินพิกัดหม้อแปลง — ย้ายเฟสแก้ไม่ได้ "
              f"(ต้องเพิ่มขนาดหม้อแปลง) ใช้เป้า %Unbalance ปกติ")
    elif T > 0:
        cap_target = (max_phase_loading_pct / T - 1.0) * 100.0 - BALANCE_MARGIN_PCT
        if cap_target < target:
            target = max(0.0, cap_target)
            goal_s = f"<{target:.0f}%" if target > 0 else "เกลี่ยเต็มที่"
            print()
            print(f"[Bottom-up Balance] โหลดรวม {T:.0f}% ของพิกัด — เป้า %Unbalance ปรับเป็น "
                  f"{goal_s} เพื่อไม่ให้เฟสใดเกิน {max_phase_loading_pct:.0f}%")
    r_raw, r_moves, r_res = _bottom_up_pass(
        net, raw, baseline, bus_to_node, min_v_thr, target_pct=target, **kw)

    ref = r_res if r_moves else baseline     # r_moves ไม่ว่าง = ผ่านเกณฑ์รับผลแล้ว
    if has_low_v and target > 0 and ref is not None and ref.low_v_nodes:
        print(f"    ยังมีโหนดแรงดันต่ำ {len(ref.low_v_nodes)} โหนด — "
              f"ลองเกลี่ยเต็มที่ (ไม่หยุดที่เกณฑ์ %Unbalance)")
        f_raw, f_moves, f_res = _bottom_up_pass(
            net, raw, baseline, bus_to_node, min_v_thr, target_pct=0.0, **kw)
        if f_moves and (len(f_res.low_v_nodes), -f_res.min_v) <                        (len(ref.low_v_nodes), -ref.min_v):
            print("    → ใช้ผลเกลี่ยเต็มที่ (แก้แรงดันได้ดีกว่า)")
            return f_raw, f_moves, f_res
        print("    → เกลี่ยเต็มที่ไม่ช่วยแรงดันเพิ่ม — ใช้ผลรอบแรก")

    return r_raw, r_moves, r_res


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — Phase Transfer Optimizer (greedy, เก็บงานแรงดันต่ำต่อจาก Step 1a)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_phase_transfer(
    net: NetworkGraph,
    raw: dict,
    baseline: SimResult,
    bus_to_node: Dict[str, int],
    min_v_thr: float,
    max_iters: int = 20,
    max_candidates_per_iter: int = 5,
    snap_tol: float = 2.0,
    max_imbalance_pct: float = DEFAULT_MAX_IMBALANCE_PCT,
    time_limit_s: float = 300.0,
    bottom_up: bool = True,
    min_gain_kw: float = MIN_MOVE_GAIN_KW,
    max_phase_loading_pct: float = DEFAULT_MAX_PHASE_LOADING_PCT,
) -> Tuple[dict, List[PhaseMove]]:
    """
    ย้ายเฟสมิเตอร์ 2 ชั้น:

      1a. Bottom-up balance — ไล่จากไลน์ branch ขึ้นมาถึงไลน์เมน จัดสมดุล kW
          ต่อเฟสด้วยการคำนวณล้วน (โยกตัวโหลดเยอะสุดก่อน) แล้ว verify 1 sim
      1b. Greedy — ไล่เก็บงานที่เหลือด้วย OpenDSS sim ต่อ 1 การย้าย
          เน้นโหนดแรงดันต่ำที่ชั้น 1a แตะไม่ถึง

    Returns (modified_raw, list_of_moves).
    """
    current_raw    = copy.deepcopy(raw)
    current_result = baseline
    moves: List[PhaseMove] = []

    # ── ชั้น 1a: ไล่สมดุลจาก branch ขึ้นเมน ────────────────────────────────
    if bottom_up:
        bu_raw, bu_moves, bu_res = optimize_phase_balance_bottom_up(
            net, current_raw, current_result, bus_to_node, min_v_thr,
            snap_tol=snap_tol, max_imbalance_pct=max_imbalance_pct,
            min_gain_kw=min_gain_kw, max_phase_loading_pct=max_phase_loading_pct)
        if bu_moves and bu_res is not None and bu_res.converged:
            current_raw    = bu_raw
            current_result = bu_res
            moves.extend(bu_moves)

    all_meters = _build_meter_inventory(net, current_raw)
    movable    = [m for m in all_meters if m.is_single_phase and m.movable_pds]

    print(f"\n[Phase Transfer] มิเตอร์ทั้งหมด={len(all_meters)}  "
          f"single-phase={sum(1 for m in all_meters if m.is_single_phase)}  "
          f"ย้ายได้={len(movable)}")

    if not movable:
        print("  ไม่มีมิเตอร์ที่ย้ายได้ — ข้าม")
        return current_raw, moves

    _t0 = time.time()
    for iteration in range(max_iters):
        if time.time() - _t0 > time_limit_s:
            print(f"  [iter {iteration}] เกิน time limit {time_limit_s:.0f}s — หยุด")
            break
        # Stop if no low-voltage nodes and imbalance acceptable
        has_problem = (bool(current_result.low_v_nodes) or
                       current_result.phase_imbalance_pct > max_imbalance_pct or
                       current_result.max_phase_loading_pct > max_phase_loading_pct)
        if not has_problem:
            print(f"  [iter {iteration}] ระบบปกติแล้ว — หยุด")
            break

        # Narrow candidates: meters near low-V nodes or on most-loaded phase
        low_v_set = set(current_result.low_v_nodes)
        neighbor_set = set(low_v_set)
        for nid in low_v_set:
            for parent in net.G.predecessors(nid):
                neighbor_set.add(parent)
            for child in net.G.successors(nid):
                neighbor_set.add(child)

        # If imbalance-only problem (no low-V), target meters on any phase that
        # carries more than the mean (not just the single heaviest) และเปิด
        # candidate pool กว้างขึ้น เพราะมิเตอร์ตัวใหญ่สุดอาจย้ายไปเฟสเบาไม่ได้
        cand_limit = max_candidates_per_iter
        light_pds: Set[int] = set()          # เฟสปลายทางที่ยอมรับ (ว่างไว้ = ทุกเฟส)
        if not low_v_set and current_result.phase_kw:
            mean_kw   = sum(current_result.phase_kw.values()) / len(current_result.phase_kw)
            heavy_pds = {NODE_TO_PD.get(n, 0)
                         for n, kw in current_result.phase_kw.items() if kw > mean_kw}
            light_pds = {NODE_TO_PD.get(n, 0)
                         for n, kw in current_result.phase_kw.items() if kw < mean_kw}
            focus = [m for m in movable if m.current_pd in heavy_pds]
            cand_limit = max(max_candidates_per_iter, 12)
        else:
            focus = [m for m in movable if m.node_id in neighbor_set]

        if not focus:
            focus = movable  # fallback: try all

        # Sort by kW descending (larger loads have more impact)
        focus.sort(key=lambda m: -m.kw)
        candidates = focus[:cand_limit]

        # เกณฑ์ขั้นต่ำ: โหมดแรงดันต่ำใช้สเกล 10×V ตามเดิม
        # โหมด imbalance ล้วน ต้องได้ imbalance ดีขึ้น ≥0.3% จริง ๆ ไม่ใช่ ΔV ระดับ noise
        low_v_mode  = bool(low_v_set)
        best_score  = 0.02 if low_v_mode else 0.30
        best_move: Optional[Tuple[MeterInfo, int, SimResult, dict]] = None

        for meter in candidates:
            for new_pd in meter.movable_pds:
                # โหมด imbalance — ย้ายเข้าเฟสที่เบากว่าค่าเฉลี่ยเท่านั้น
                # (ย้ายจากเฟสหนักไปเฟสหนักด้วยกันไม่ช่วยอะไร แต่กิน sim ฟรี)
                if light_pds and new_pd not in light_pds:
                    continue
                candidate_raw = copy.deepcopy(current_raw)
                _apply_meter_phase(candidate_raw["features"], meter.feat_idx, new_pd)

                result = _sim_modified_json(
                    candidate_raw, bus_to_node,
                    net.rated_kva, min_v_thr, snap_tol)
                score = _score_result(current_result, result,
                                      low_v_mode=low_v_mode, min_v_thr=min_v_thr,
                                      max_phase_loading_pct=max_phase_loading_pct)

                if score > best_score:
                    best_score = score
                    best_move  = (meter, new_pd, result, candidate_raw)

        if best_move is None:
            print(f"  [iter {iteration+1}] ไม่มีการย้ายที่ดีกว่า — หยุด")
            break

        meter, new_pd, result, candidate_raw = best_move
        from_pd = meter.current_pd
        dv      = result.min_v - current_result.min_v
        di      = current_result.phase_imbalance_pct - result.phase_imbalance_pct

        moves.append(PhaseMove(meter=meter, from_pd=from_pd, to_pd=new_pd,
                               delta_min_v=dv, delta_imbalance=di))
        print(f"  [iter {iteration+1}] {meter.peano or 'idx='+str(meter.feat_idx):>14}  "
              f"{PD_TO_PHASE.get(from_pd,'?')}→{PD_TO_PHASE.get(new_pd,'?')}  "
              f"ΔV={dv:+.1f}V  Δimbal={di:+.1f}%  Vmin={result.min_v:.1f}V")

        current_result   = result
        current_raw      = candidate_raw
        meter.current_pd = new_pd  # update in-memory

    return current_raw, moves


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Conductor Upgrade Simulator
# ─────────────────────────────────────────────────────────────────────────────

def simulate_conductor_upgrade(
    net: NetworkGraph,
    raw: dict,
    baseline: SimResult,
    bus_to_node: Dict[str, int],
    min_v_thr: float,
    snap_tol: float = 2.0,
) -> List[UpgradeResult]:
    """
    For each low-voltage subtree, upgrade conductor size (→ 95 mm²) on EVERY
    segment of the path from the transformer to each low-V node in that subtree
    (union of paths), not just the single entry segment — voltage drop is
    cumulative along the feeder, so a one-segment upgrade rarely helps.
    Only segments currently < 95 mm² are upgraded. One UpgradeResult per subtree
    (deduplicated: a subtree covered by an ancestor entry edge is skipped).
    Returns list sorted by delta_min_v descending.
    """
    results: List[UpgradeResult] = []
    low_v_set = set(baseline.low_v_nodes)

    if not low_v_set:
        print("\n[Conductor Upgrade] ไม่พบโหนดแรงดันต่ำ — ข้าม")
        return []

    # Find edges (u→v) in BFS tree where v's subtree contains low-V nodes
    # Deduplicate: skip subtrees already covered by an ancestor edge
    processed: Set[int] = set()
    candidate_edges: List[Tuple[int, int]] = []

    # BFS order from transformer so we process shallow edges first
    for (u, v) in nx.bfs_edges(net.G, net.transformer_node):
        if v in processed:
            continue
        subtree_v = net.get_subtree((u, v)) | {v}
        if subtree_v & low_v_set:
            candidate_edges.append((u, v))
            processed.update(subtree_v)

    # ── เพิ่มขนาดสายเฉพาะกรณีมีมิเตอร์ EV charger ─────────────────────────
    ev_meters: List[Tuple[int, int, str, float]] = []   # (feat_idx, node, peano, kw)
    for fi, nid in net.load_feat_nodes.items():
        kw = float(get_attr(raw['features'][fi], 'KWP', 0.0) or 0.0)
        if _is_ev_kw(kw):
            peano = str(get_attr(raw['features'][fi], 'PEANO', '') or '').strip()
            ev_meters.append((fi, nid, peano, kw))
    ev_nodes = {e[1] for e in ev_meters}

    if not ev_meters:
        print()
        print(f'[Conductor Upgrade] ไม่พบมิเตอร์ EV charger (KWP {EV_KW_MIN:.1f}-{EV_KW_MAX:.1f} kW) '
              f'— ข้าม (เพิ่มขนาดสายเฉพาะกรณีมี EV)')
        return []

    candidate_edges = [(u, v) for (u, v) in candidate_edges
                       if (net.get_subtree((u, v)) | {v}) & ev_nodes]
    if not candidate_edges:
        print()
        print(f'[Conductor Upgrade] มี EV {len(ev_meters)} ตัว แต่ไม่อยู่ในพื้นที่แรงดันต่ำ — ข้าม')
        return []

    print(f"\n[Conductor Upgrade] ทดสอบ {len(candidate_edges)} subtree "
          f"(อัปเกรดทั้ง path หม้อแปลง→โหนดแรงดันต่ำ)")

    max_size = _AW_SIZES[-1]

    for (u, v) in candidate_edges:
        subtree      = net.get_subtree((u, v)) | {v}
        lowv_in_sub  = subtree & low_v_set
        n_affected   = len(lowv_in_sub)

        # รวม edge บน path หม้อแปลง→โหนดแรงดันต่ำแต่ละตัวใน subtree นี้ (union ของ path)
        path_edges: Dict[int, Tuple[int, int]] = {}
        for target in lowv_in_sub:
            for fi, pu, pv in _path_to_node(net, target):
                path_edges[fi] = (pu, pv)

        # เฉพาะช่วงที่กระแส EV ไหลผ่าน ร่วมกับ path ไปโหนดแรงดันต่ำ
        # (แรงดันตกที่โหนด t เพราะ EV = I_ev x Z ของช่วงสายที่ใช้ร่วมกัน)
        evs_here = [e for e in ev_meters if e[1] in subtree]
        ev_path: Set[int] = set()
        for _efi, e_node, _ep, _ekw in evs_here:
            for fi, _pu, _pv in _path_to_node(net, e_node):
                ev_path.add(fi)
        path_edges = {fi: uv for fi, uv in path_edges.items() if fi in ev_path}

        if not path_edges:
            print(f"  subtree ({u}→{v}): หา path ไม่ได้ — ข้าม")
            continue

        # เลือกเฉพาะ segment ที่ยังเล็กกว่าขนาดใหญ่สุดใน AW_IMP
        from_sizes: Dict[int, int] = {}
        for fi in path_edges:
            cur = int(get_attr(raw["features"][fi], "CONDUCTORSIZE", 50) or 50)
            if cur < max_size:
                from_sizes[fi] = cur

        if not from_sizes:
            print(f"  subtree ({u}→{v}): ทุก segment ≥ {max_size}mm² แล้ว — ข้าม")
            continue

        # อัปเกรดหนึ่งสเต็ปจากขนาดใหญ่สุดที่พบบน path (ทุก segment เท่ากันหมด)
        cur_max     = max(from_sizes.values())
        next_sizes  = [s for s in _AW_SIZES if s > cur_max]
        target_size = next_sizes[0] if next_sizes else max_size
        to_upgrade  = [(fi, path_edges[fi][0], path_edges[fi][1]) for fi in from_sizes]

        candidate_raw = copy.deepcopy(raw)
        for fi, _, _ in to_upgrade:
            candidate_raw["features"][fi].setdefault("attributes", {})["CONDUCTORSIZE"] = target_size

        result = _sim_modified_json(candidate_raw, bus_to_node,
                                    net.rated_kva, min_v_thr, snap_tol)
        delta_v = result.min_v - baseline.min_v if result.converged else -999.0

        status = (f"Vmin={result.min_v:.1f}V  ΔV={delta_v:+.1f}V"
                  if result.converged else f"[NG] {result.error}")
        print(f"  subtree ({u}→{v})  {len(to_upgrade)} segment →{target_size}mm²  "
              f"{n_affected}โหนด LowV  {status}")

        print('      EV: ' + ', '.join(f'{e[2] or e[0]} ({e[3]:.1f}kW)' for e in evs_here))
        entry_fi = net.find_switch_edge_feature((u, v))
        if entry_fi is None:
            entry_fi = to_upgrade[0][0]
        results.append(UpgradeResult(
            edge=(u, v), feat_idx=entry_fi,
            from_size=from_sizes.get(entry_fi, cur_max), to_size=target_size,
            n_affected=n_affected, result=result, delta_min_v=delta_v,
            upgraded_edges=to_upgrade, from_sizes=from_sizes,
            ev_peanos=[e[2] or f"idx={e[0]}" for e in evs_here],
        ))

    results.sort(key=lambda r: -r.delta_min_v)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Phase Addition Simulator
# ─────────────────────────────────────────────────────────────────────────────

def _feat_length_m(feat: dict) -> float:
    """ความยาวสาย 1 segment (m) — คิดจาก geometry, ถ้าไม่มีใช้ MEASURELENGTH"""
    tot = 0.0
    for path in (feat.get("geometry") or {}).get("paths") or []:
        for p1, p2 in zip(path, path[1:]):
            tot += math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
    if tot <= 0.0:
        tot = float(get_attr(feat, "MEASURELENGTH", 0.0) or 0.0)
    return tot


def build_non_full_phase_comps(
    net: NetworkGraph,
    raw: dict,
    tx_mask: int,
    min_len_m: float = DESIGN_MIN_COMP_LEN_M,
    min_meters: int = DESIGN_MIN_COMP_METERS,
) -> List[DesignComp]:
    """
    จัดกลุ่มสาย LV ที่ "เฟสไม่ครบตามหม้อแปลง" เป็น connected component

    ต้องเพิ่มเฟสทั้งกลุ่มพร้อมกันเสมอ — อัปเกรดกลางสายไม่ได้ (จะเกิด A → ABC → A)
    กลุ่มที่ยาว ≥ min_len_m หรือมีมิเตอร์ ≥ min_meters ถือว่าผิดหลักออกแบบ (violates)
    เรียงจากกลุ่มใหญ่ (ยาวมาก) ไปเล็ก
    """
    g: nx.Graph = nx.Graph()
    info: Dict[int, Tuple[int, int, int]] = {}      # feat_idx → (u, v, pd) ทิศตาม BFS

    for feat_idx, (pu, pv) in net.lc_feat_edges.items():
        pd = int(get_attr(raw["features"][feat_idx], "PHASEDESIGNATION", 7) or 7)
        if (pd & tx_mask) == tx_mask:
            continue                                # ครบเฟสหม้อแปลงแล้ว
        if net.G.has_edge(pu, pv):
            u, v = pu, pv
        elif net.G.has_edge(pv, pu):
            u, v = pv, pu
        else:
            continue
        g.add_edge(u, v)
        info[feat_idx] = (u, v, pd)

    comps: List[DesignComp] = []
    for nodes in nx.connected_components(g):
        edges = [(fi, u, v, pd) for fi, (u, v, pd) in info.items()
                 if u in nodes and v in nodes]
        if not edges:
            continue
        length = sum(_feat_length_m(raw["features"][fi]) for fi, _u, _v, _pd in edges)
        meter_fis = [fi for fi, nid in net.load_feat_nodes.items() if nid in nodes]
        kw = sum(float(get_attr(raw["features"][fi], "KWP", 0.0) or 0.0) for fi in meter_fis)
        comps.append(DesignComp(
            idx=0, nodes=set(nodes), edges=edges, length_m=length,
            n_meters=len(meter_fis), kw=kw, pds={pd for _fi, _u, _v, pd in edges},
            violates=(length >= min_len_m or len(meter_fis) >= min_meters),
        ))

    comps.sort(key=lambda c: -c.length_m)
    for i, c in enumerate(comps):
        c.idx = i
    return comps


def assess_design(
    net: NetworkGraph,
    raw: dict,
    tx_mask: int,
    min_len_m: float = DESIGN_MIN_COMP_LEN_M,
    min_meters: int = DESIGN_MIN_COMP_METERS,
) -> Tuple[List[DesignComp], Dict[str, float]]:
    """
    ตรวจมาตรฐานเฟสสายของทั้งหม้อแปลง

    Returns (comps, summary) — summary มี:
      len_total / len_partial / ratio_full_pct  (สัดส่วนความยาวสายที่ครบเฟสหม้อแปลง)
      kw_total  / kw_partial  / kw_partial_pct  (โหลดที่อยู่บนสายไม่ครบเฟส)
      n_comps / n_violate / len_violate / meters_violate
    """
    comps = build_non_full_phase_comps(net, raw, tx_mask, min_len_m, min_meters)

    len_total = sum(_feat_length_m(raw["features"][fi]) for fi in net.lc_feat_edges)
    len_partial = sum(c.length_m for c in comps)
    kw_total = sum(float(get_attr(raw["features"][fi], "KWP", 0.0) or 0.0)
                   for fi in net.load_feat_nodes)
    kw_partial = sum(c.kw for c in comps)
    viol = [c for c in comps if c.violates]

    summary = dict(
        len_total=len_total, len_partial=len_partial,
        ratio_full_pct=(100.0 * (len_total - len_partial) / len_total) if len_total > 0 else 100.0,
        kw_total=kw_total, kw_partial=kw_partial,
        kw_partial_pct=(100.0 * kw_partial / kw_total) if kw_total > 0 else 0.0,
        n_comps=len(comps), n_violate=len(viol),
        len_violate=sum(c.length_m for c in viol),
        meters_violate=sum(c.n_meters for c in viol),
    )
    return comps, summary


def _pa_score(r: PhaseAddResult) -> float:
    """คะแนนรวมของ Phase Addition — Vmin สำคัญกว่า imbalance 10 เท่า (เหมือน _score_result)."""
    if not r.result.converged:
        return -999.0
    return r.delta_min_v * 10.0 + r.delta_imbalance


def simulate_phase_addition(
    net: NetworkGraph,
    raw: dict,
    baseline: SimResult,
    bus_to_node: Dict[str, int],
    min_v_thr: float,
    snap_tol: float = 2.0,
    max_imbalance_pct: float = DEFAULT_MAX_IMBALANCE_PCT,
    max_test_comps: int = 6,
    max_phase_loading_pct: float = DEFAULT_MAX_PHASE_LOADING_PCT,
    design_min_len_m: float = DESIGN_MIN_COMP_LEN_M,
    design_min_meters: int = DESIGN_MIN_COMP_METERS,
    design_check: bool = True,
    prefer_design: bool = False,
) -> List[PhaseAddResult]:
    """
    Upgrade LV line components that lack a transformer phase to ALL transformer
    phases (3-phase TX -> ABC, 2-phase TX -> its 2 phases) and redistribute
    their meters across A/B/C by kW, then simulate.

    ทำงาน 3 โหมด:
      • low-V mode      — มีโหนดแรงดันต่ำ → เล็ง component ที่มีโหนด low-V
      • imbalance mode  — ไม่มี low-V แต่ phase imbalance > max_imbalance_pct
                          → เล็ง component ที่แบกโหลดมิเตอร์มากสุด (ตัวการ imbalance
                          เพราะมิเตอร์ติดอยู่บนสายที่ไม่ครบ 3 เฟส ย้ายเฟสไม่ได้)
      • design mode     — ไฟฟ้าผ่านเกณฑ์หมดแล้ว แต่ยังมีกลุ่มสายที่ไม่ครบเฟสหม้อแปลง
                          และใหญ่พอ (ยาว ≥ design_min_len_m หรือมิเตอร์ ≥ design_min_meters)
                          → เล็งกลุ่มเหล่านั้นทั้งหมด ตามหลักออกแบบระบบจำหน่าย

    Returns list sorted by _pa_score descending (Vmin ก่อน, แล้ว imbalance);
    design mode เรียงตามจำนวนกลุ่มที่แก้ให้ครบเฟสได้ก่อน
    """
    results: List[PhaseAddResult] = []
    low_v_set = set(baseline.low_v_nodes)
    imbalance_mode = (not low_v_set and
                      (baseline.phase_imbalance_pct > max_imbalance_pct or
                       baseline.max_phase_loading_pct > max_phase_loading_pct))

    # ── สร้าง undirected graph ของ edges ที่ไม่ใช่ 3 เฟส ──────────────────
    # ต้องอัปเกรดทั้ง connected component พร้อมกัน
    # (ไม่สามารถอัปเกรดกลางสายได้ เพราะจะเกิด A→ABC→A)
    # ── เฟสที่หม้อแปลงมีจริง — เพิ่มเฟสสายได้ไม่เกินเฟสของหม้อแปลง ───────────
    # หม้อแปลง 2 เฟส → เพิ่มสายเฟสเดียวเป็น 2 เฟสตามหม้อแปลง ไม่ใช่ ABC
    # (ไม่งั้นมิเตอร์ที่ย้ายไปเฟสที่หม้อแปลงไม่มีจะไฟดับจริง และผลจำลองจะเห็นโหลดหายไปแบบหลอก ๆ)
    tx_pds  = _tx_available_pds(baseline)
    tx_mask = 0
    for _tpd in tx_pds:
        tx_mask |= _tpd
    if len(tx_pds) < 2:
        print()
        print("[Phase Addition] หม้อแปลงมีเฟสเดียว — เพิ่มเฟสสายไม่ได้ "
              "(ต้องเพิ่มเฟสหม้อแปลง) — ข้าม")
        return []

    # ── กลุ่มสายที่ไม่ครบเฟสหม้อแปลง (ต้องอัปเกรดทั้งกลุ่มพร้อมกัน) ──────────
    design_comps = build_non_full_phase_comps(
        net, raw, tx_mask, design_min_len_m, design_min_meters)
    all_comps: List[Tuple[Set[int], List]] = [(c.nodes, c.edges) for c in design_comps]
    violating = [c for c in design_comps if c.violates] if design_check else []

    # design mode — ไฟฟ้าผ่านหมดแล้ว แต่ยังมีสายไม่ครบเฟสที่ใหญ่พอตามหลักออกแบบ
    design_mode = (not low_v_set) and (not imbalance_mode) and bool(violating)

    if not low_v_set and not imbalance_mode and not design_mode:
        print("\n[Phase Addition] ไม่มีโหนดแรงดันต่ำ, imbalance ไม่เกินเกณฑ์ "
              "และสายครบเฟสตามหลักออกแบบ — ข้าม")
        return []

    # ── เลือกเฉพาะ component ที่มี low-V node ────────────────────────────────
    # แต่ละ connected component เริ่มจาก junction กับสาย 3-phase (tab แยก)
    # ไปถึงปลายทาง — entry edge เป็น 3-phase อยู่แล้วโดย definition ของ component
    # ไม่ต้องตรวจ upstream: ถ้าสาย non-3-phase ต่อกันโดยตรงมันจะอยู่ใน component เดียวกัน
    def _comp_meter_kw(comp_nodes: Set[int]) -> float:
        return sum(
            float(get_attr(raw["features"][fi], "KWP", 0.0) or 0.0)
            for fi, nid in net.load_feat_nodes.items() if nid in comp_nodes
        )

    if low_v_set:
        target_cis = [ci for ci, (nodes, _) in enumerate(all_comps)
                      if nodes & low_v_set]
        mode_str = "low-V"
        if not target_cis:
            print("\n[Phase Addition] ไม่พบ component ที่มีโหนด low-V — ข้าม")
            return []
    elif imbalance_mode:
        # imbalance mode — เรียง component ตามโหลดมิเตอร์ที่แบกอยู่ (มาก→น้อย)
        # component ที่มีโหลดมากบนสายไม่ครบเฟส คือ ตัวการหลักของ imbalance
        ranked = sorted(
            ((ci, _comp_meter_kw(nodes)) for ci, (nodes, _) in enumerate(all_comps)),
            key=lambda x: -x[1],
        )
        target_cis = [ci for ci, kw in ranked if kw > 0][:max_test_comps]
        mode_str = f"imbalance {baseline.phase_imbalance_pct:.0f}%>{max_imbalance_pct:.0f}%"
        if not target_cis:
            print("\n[Phase Addition] ไม่มี component ที่แบกโหลดมิเตอร์ — ข้าม")
            return []
    else:
        # design mode — ไฟฟ้าผ่านเกณฑ์แล้ว แต่สายยังไม่ครบเฟสตามหลักออกแบบ
        # เล็งทุกกลุ่มที่เข้าเกณฑ์ (กลุ่มใหญ่ก่อน) — เป้าหมายคือทำให้ครบเฟส ไม่ใช่แก้ตัวเลข
        target_cis = [c.idx for c in violating][:max_test_comps]
        mode_str = (f"design {len(violating)} กลุ่มไม่ครบเฟส "
                    f"(≥{design_min_len_m:.0f}m หรือ ≥{design_min_meters} มิเตอร์)")

    # ── Build test cases ─────────────────────────────────────────────────
    # tuple: (nodes, edges, is_all, [], orig_comp_idx)
    test_cases: List[Tuple[Set[int], List, bool, List[int], int]] = []
    for ci in target_cis:
        comp_nodes, comp_edges = all_comps[ci]
        test_cases.append((comp_nodes, comp_edges, False, [], ci))

    # ตัวเลือก "รวมทุกกลุ่มเป้าหมาย"
    # design mode รวมทุกกลุ่มที่ผิดหลักออกแบบ (ไม่ตัดที่ max_test_comps)
    # เพราะหน้างานเดินสายเพิ่มเฟสทีเดียวทั้งหม้อแปลง ไม่ได้ทำทีละกลุ่ม
    combined_cis = ([c.idx for c in violating]
                    if (design_mode or (prefer_design and violating)) else target_cis)
    if len(combined_cis) > 1:
        comb_nodes: Set[int] = set()
        comb_edges: List = []
        for ci in combined_cis:
            comb_nodes |= all_comps[ci][0]
            comb_edges.extend(all_comps[ci][1])
        test_cases.append((comb_nodes, comb_edges, True, [], -1))

    n_all = len(all_comps)
    print(f"\n[Phase Addition] โหมด {mode_str}  พบ {n_all} กลุ่ม non-3-phase  "
          f"เล็ง {len(target_cis)} กลุ่ม  (ทดสอบ {len(test_cases)} ตัวเลือก)")

    def _run_comp(comp_nodes, comp_edges, label):
        n_affected = len(comp_nodes & low_v_set)
        rep_fi, rep_u, rep_v, rep_pd = comp_edges[0]

        # Redistribute ทุกมิเตอร์ใน component (ทุก node มี A/B/C หลัง upgrade)
        meters_here: List[Tuple[int, float]] = []
        for fi, nid in net.load_feat_nodes.items():
            if nid in comp_nodes:
                m_pd = _meter_current_pd(raw["features"][fi])
                if m_pd in SINGLE_PDS:
                    kw = float(get_attr(raw["features"][fi], "KWP", 0.0) or 0.0)
                    meters_here.append((fi, kw))

        # กระจายเฟสโดย "รู้ว่าส่วนที่เหลือของเครือข่ายเอียงไปทางไหนอยู่แล้ว"
        # seed ด้วยโหลดนอก component (baseline ที่หม้อแปลง หักส่วนที่ component นี้แบก)
        # ไม่งั้นจะแบ่งใน component ให้เท่ากัน 1/3 ทั้งที่เป้าหมายคือหม้อแปลงสมดุล
        _here = {fi for fi, _ in meters_here}
        phase_kw_used = {pd: 0.0 for pd in tx_pds}
        for _n, _kw in baseline.phase_kw.items():
            _pd = NODE_TO_PD.get(_n)
            if _pd in phase_kw_used:
                phase_kw_used[_pd] = max(0.0, _kw)
        for fi, kw in meters_here:                    # หักโหลดของ component ออก
            _pd = _meter_current_pd(raw["features"][fi])
            if _pd in phase_kw_used:
                phase_kw_used[_pd] = max(0.0, phase_kw_used[_pd] - kw)

        # ── กระจายเฟสแบบ "ทุกกิ่งเอียงเท่ากัน" ────────────────────────────
        # เดิมกระจายเป็นกองเดียวทั้งหม้อแปลง → หม้อแปลงสมดุล แต่กิ่งหนึ่งอาจหนัก B
        # อีกกิ่งหนัก A หักล้างกันพอดี (กระแสนิวทรัลในกิ่งสูง)
        #
        # ตอนนี้: รวบมิเตอร์เป็น "กอง" ตามชุมสาย (แบบเดียวกับ bottom-up) แล้วให้ทุกกอง
        # เล็งสัดส่วนเฟสเดียวกัน = สัดส่วนที่ทำให้หม้อแปลงสมดุลพอดีเมื่อรวมกับโหลดนอกกลุ่ม
        #   target[p] = (โหลดรวมทั้งหมด/จำนวนเฟส) − โหลดนอกกลุ่มของเฟสนั้น
        # ถ้าโหลดนอกกลุ่มสมดุลอยู่แล้ว สัดส่วนนี้จะเท่ากันทุกเฟส = แต่ละกิ่งสมดุลด้วย
        # ถ้าโหลดนอกกลุ่มเอียง ทุกกิ่งจะเอียงเท่า ๆ กันคนละนิด แทนที่จะเอียงหนักกองเดียว
        kw_of = dict(meters_here)
        at_node: Dict[int, List[int]] = defaultdict(list)
        for fi, nid in net.load_feat_nodes.items():
            if fi in kw_of and nid in comp_nodes:
                at_node[nid].append(fi)

        # โหนดบนสุดของกลุ่ม = โหนดที่พ่อไม่ได้อยู่ในกลุ่ม
        roots = [n for n in comp_nodes
                 if not any(p in comp_nodes for p in net.G.predecessors(n))]
        order: List[int] = []
        seen_n: Set[int] = set()
        dq = deque(roots or sorted(comp_nodes)[:1])
        while dq:
            u = dq.popleft()
            if u in seen_n:
                continue
            seen_n.add(u)
            order.append(u)
            for c in net.G.successors(u):
                if c in comp_nodes and c not in seen_n:
                    dq.append(c)
        for n in comp_nodes:                    # กันโหนดที่หลุดจาก BFS
            if n not in seen_n:
                order.append(n)

        # สัดส่วนเป้าหมายของแต่ละเฟส — ใช้ชุดเดียวกันทุกชั้นของต้นไม้
        comp_kw = sum(kw_of.values())
        ideal = (sum(phase_kw_used.values()) + comp_kw) / len(tx_pds)
        need = {pd: max(0.0, ideal - phase_kw_used[pd]) for pd in tx_pds}
        need_tot = sum(need.values())
        if need_tot <= 0:                       # โหลดนอกกลุ่มเอียงเกินกว่าจะชดเชยได้
            share = {pd: 1.0 / len(tx_pds) for pd in tx_pds}
        else:
            share = {pd: need[pd] / need_tot for pd in tx_pds}

        # ── รวบมิเตอร์เป็นกองตามชุมสาย (เหมือน bottom-up) ────────────────────
        kids_of = {v: [c for c in net.G.successors(v) if c in comp_nodes] for v in order}
        pools: List[List[int]] = []
        pend_at: Dict[int, List[int]] = defaultdict(list)
        for v in reversed(order):
            pend = list(at_node.get(v, []))
            for c in kids_of[v]:
                pend.extend(pend_at.pop(c, []))
            # flush ที่ชุมสาย / ที่โหนดบนสุด / หรือเมื่อกองใหญ่พอจะแบ่งเฟสได้เอง
            # (ไม่งั้นลูกซอยยาวกลางทางจะถูกยกไปรวมที่ชุมสาย แล้วเหลือแค่ 1-2 เฟสทั้งซอย)
            if pend and (len(kids_of[v]) >= 2 or v in roots or
                         len(pend) >= _POOL_FLUSH_N or
                         not any(p in comp_nodes for p in net.G.predecessors(v))):
                pools.append(pend)
            else:
                pend_at[v] = pend
        for rest in pend_at.values():
            if rest:
                pools.append(rest)

        # ── เริ่มจาก "เฟสเดิม" แล้วย้ายเท่าที่จำเป็น ──────────────────────────
        # เพิ่มเฟสสายแล้วไม่ได้แปลว่าต้องย้ายมิเตอร์ทุกตัว — ย้ายตัวโหลดมากก่อน
        # และหยุดทันทีที่กองนั้นเบี่ยงเบนจากเป้าต่ำกว่าเกณฑ์ (ประหยัดงานหน้างาน)
        target_pct = max(1.0, max_imbalance_pct - BALANCE_MARGIN_PCT)
        assign_now: Dict[int, int] = {}
        for fi, _kw in meters_here:
            cur = _meter_current_pd(raw["features"][fi])
            assign_now[fi] = cur if cur in tx_pds else 0      # 0 = เฟสเดิมใช้ไม่ได้

        def _dev_pct(acc: Dict[int, float], tgt: Dict[int, float], tot: float) -> float:
            """เบี่ยงเบนจากเป้า คิดเป็น % ของโหลดเฉลี่ยต่อเฟส (นิยามเดียวกับ %Unbalance)"""
            if tot <= 0:
                return 0.0
            mean_v = tot / len(tx_pds)
            return max(abs(acc[p] - tgt[p]) for p in tx_pds) / mean_v * 100.0

        def _balance_pool(fis: List[int], glob_mode: bool = False) -> None:
            # เป้าของกอง = แบ่งเท่ากันทุกเฟสของหม้อแปลง
            # (ไม่ใช้ share ที่เอียงตามโหลดนอกกลุ่ม — ไม่งั้นเฟสที่เพิ่งเดินเพิ่มจะไม่มีโหลด
            #  ขึ้นเลยทั้งกิ่ง ซึ่งผิดวัตถุประสงค์ของการเพิ่มเฟส) ความเอียงที่หม้อแปลง
            # ค่อยไปแก้ในรอบระดับหม้อแปลงด้านล่าง โดยย้ายเพิ่มเท่าที่จำเป็น
            tot = sum(kw_of[f] for f in fis)
            if tot <= 0:
                return
            even = 1.0 / len(tx_pds)
            tgt = {p: tot * ((1.0 - _POOL_SHARE_W) * even + _POOL_SHARE_W * share[p])
                   for p in tx_pds}
            acc = {p: 0.0 for p in tx_pds}
            for f in fis:
                acc[assign_now[f]] += kw_of[f]
            # ย้ายตัวโหลดมากก่อน หยุดเมื่อกองนี้เข้าเกณฑ์ หรือหม้อแปลงเข้าเกณฑ์แล้ว
            for f in sorted(fis, key=lambda x: -kw_of[x]):
                if (_dev_pct(acc, tgt, tot) < target_pct or
                        _spread_pct(phase_kw_used) < target_pct):
                    break
                kw, cur = kw_of[f], assign_now[f]
                if kw <= 0:
                    continue
                best_p, best_gain = cur, MIN_MOVE_GAIN_KW
                base = _rms_kw({p: acc[p] - tgt[p] for p in tx_pds})
                for p in tx_pds:
                    if p == cur:
                        continue
                    acc[cur] -= kw; acc[p] += kw
                    gain = base - _rms_kw({q: acc[q] - tgt[q] for q in tx_pds})
                    acc[cur] += kw; acc[p] -= kw
                    if gain > best_gain:
                        best_gain, best_p = gain, p
                if best_p != cur:
                    acc[cur] -= kw; acc[best_p] += kw
                    phase_kw_used[cur] -= kw; phase_kw_used[best_p] += kw
                    assign_now[f] = best_p

        # รวมโหลดของกลุ่ม (ณ เฟสเดิม) เข้ากับโหลดนอกกลุ่ม → phase_kw_used = ภาพรวมสด
        # มิเตอร์ที่เฟสเดิมหม้อแปลงไม่มี ต้องเลือกเฟสใหม่ตั้งแต่ตอนนี้
        for fi, _kw in sorted(meters_here, key=lambda x: -x[1]):
            if not assign_now[fi]:
                assign_now[fi] = min(tx_pds, key=lambda q: phase_kw_used[q])
            phase_kw_used[assign_now[fi]] += kw_of[fi]

        # ── เกลี่ยกิ่งที่เพิ่งเพิ่มเฟส (กองใหญ่ก่อน) จนหม้อแปลงเข้าเกณฑ์ ──────
        for pool in sorted(pools, key=lambda pl: -sum(kw_of[f] for f in pl)):
            if _spread_pct(phase_kw_used) < target_pct:
                break                       # %Unbalance ถึงเกณฑ์แล้ว — ไม่ต้องย้ายต่อ
            _balance_pool(pool)

        # ถ้ายังไม่เข้าเกณฑ์ ค่อยเกลี่ยระดับหม้อแปลง (ข้ามขอบเขตกอง)
        if _spread_pct(phase_kw_used) >= target_pct:
            all_fis = [fi for fi, _ in sorted(meters_here, key=lambda x: -x[1])]
            for fi in all_fis:
                if _spread_pct(phase_kw_used) < target_pct:
                    break
                kw, cur = kw_of[fi], assign_now[fi]
                if kw <= 0:
                    continue
                best_p, best_gain = cur, MIN_MOVE_GAIN_KW
                base = _rms_kw(phase_kw_used)
                for p in tx_pds:
                    if p == cur:
                        continue
                    phase_kw_used[cur] -= kw; phase_kw_used[p] += kw
                    gain = base - _rms_kw(phase_kw_used)
                    phase_kw_used[cur] += kw; phase_kw_used[p] -= kw
                    if gain > best_gain:
                        best_gain, best_p = gain, p
                if best_p != cur:
                    phase_kw_used[cur] -= kw; phase_kw_used[best_p] += kw
                    assign_now[fi] = best_p

        # เก็บเฉพาะตัวที่เฟสเปลี่ยนจริง = งานที่ต้องทำหน้างาน
        meter_moves = [(fi, pd) for fi, pd in assign_now.items()
                       if pd != _meter_current_pd(raw["features"][fi])]

        candidate_raw = copy.deepcopy(raw)
        for fi, eu, ev, _ in comp_edges:
            candidate_raw["features"][fi].setdefault("attributes", {})["PHASEDESIGNATION"] = tx_mask
        for fi, new_m_pd in meter_moves:
            _apply_meter_phase(candidate_raw["features"], fi, new_m_pd)

        result  = _sim_modified_json(candidate_raw, bus_to_node,
                                     net.rated_kva, min_v_thr, snap_tol)
        if result.converged:
            delta_v = result.min_v - baseline.min_v
            delta_i = baseline.phase_imbalance_pct - result.phase_imbalance_pct
            status  = (f"Vmin={result.min_v:.1f}V  ΔV={delta_v:+.1f}V  "
                       f"Imbal={result.phase_imbalance_pct:.1f}%  Δimbal={delta_i:+.1f}%  "
                       f"ย้าย {len(meter_moves)} มิเตอร์")
        else:
            delta_v, delta_i = -999.0, -999.0
            status = f"[NG] {result.error}"
        # กลุ่มที่ผิดหลักออกแบบซึ่งตัวเลือกนี้ทำให้ครบเฟส (ต้องครอบคลุมทั้งกลุ่ม)
        _fixed = sum(1 for c in violating if c.nodes <= comp_nodes)
        _ncomp = sum(1 for c in design_comps if c.nodes <= comp_nodes)
        print(f"  {label}  {n_affected}LowV  {status}"
              + (f"  แก้สายไม่ครบเฟส {_fixed}/{len(violating)} กลุ่ม" if violating else ""))
        return PhaseAddResult(
            feat_idx=rep_fi, edge=(rep_u, rep_v),
            from_pd=rep_pd, to_pd=tx_mask,
            n_affected=n_affected, meter_moves=meter_moves,
            upgraded_edges=[(fi, eu, ev) for fi, eu, ev, _ in comp_edges],
            result=result, delta_min_v=delta_v, delta_imbalance=delta_i,
            n_comps=max(1, _ncomp), design_fixed=_fixed,
        )

    for i, (comp_nodes, comp_edges, is_all, dep_cis, orig_ci) in enumerate(test_cases, 1):
        if is_all:
            label = (f"[รวม {len(all_comps)} กลุ่ม] "
                     f"({len(comp_nodes)} nodes, {len(comp_edges)} edges)")
        elif dep_cis:
            dep_str = "+".join(str(d + 1) for d in dep_cis)
            label = (f"[กลุ่ม {orig_ci+1}+ต้นน้ำ{dep_str}] "
                     f"({len(comp_nodes)} nodes, {len(comp_edges)} edges)")
        else:
            label = f"[กลุ่ม {orig_ci+1}] ({len(comp_nodes)} nodes, {len(comp_edges)} edges)"
        results.append(_run_comp(comp_nodes, comp_edges, label))

    if design_mode or (prefer_design and violating):
        # เป้าหมายคือ "ทำให้ครบเฟสตามหลักออกแบบ" — เรียงตามจำนวนกลุ่มที่แก้ได้ก่อน
        # (ตัวเลือกรวมทุกกลุ่มจึงมาเป็นอันดับ 1) แล้วค่อยดู imbalance / Vmin
        results.sort(key=lambda r: (
            -(r.design_fixed if r.result.converged else -1),
            -(r.delta_imbalance if r.result.converged else -999.0),
            -r.delta_min_v,
        ))
    elif imbalance_mode:
        # เป้าหมายหลักคือลด imbalance — เรียงตาม Δimbal ก่อน แล้วค่อย Vmin
        results.sort(key=lambda r: (
            -(r.delta_imbalance if r.result.converged else -999.0),
            -r.delta_min_v,
        ))
    else:
        results.sort(key=lambda r: -_pa_score(r))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class LVOptimizer:
    """
    Analyse one LV transformer network and produce improvement recommendations.

    Run order:
      1. Fetch/parse JSON  →  2. Baseline DSS  →  3. Phase transfer
      →  4. Conductor upgrade (if still problems)
      →  5. Phase addition   (if still problems)
    """

    def __init__(
        self,
        facilityid: str,
        min_voltage_v: float = 200.0,
        max_imbalance_pct: float = DEFAULT_MAX_IMBALANCE_PCT,
        json_dir: str = JSON_DIR,
        max_sim_time_s: float = 300.0,
        min_move_gain_kw: float = MIN_MOVE_GAIN_KW,
        max_phase_loading_pct: float = DEFAULT_MAX_PHASE_LOADING_PCT,
        design_check: bool = True,
        design_min_len_m: float = DESIGN_MIN_COMP_LEN_M,
        design_min_meters: int = DESIGN_MIN_COMP_METERS,
        region: Optional[str] = None,
    ) -> None:
        self.facilityid        = facilityid
        self.min_voltage_v     = min_voltage_v
        self.max_imbalance_pct = max_imbalance_pct
        self.json_dir          = json_dir
        self.max_sim_time_s    = max_sim_time_s
        self.min_move_gain_kw  = min_move_gain_kw
        self.max_phase_loading_pct = max_phase_loading_pct
        self.design_check      = design_check
        self.design_min_len_m  = design_min_len_m
        self.design_min_meters = design_min_meters
        # region: ใช้เลือก GIS server ปลายทางตอนดึง JSON (multi-region rollout)
        self.region            = region

        # error: ให้ run_web.py ใช้แสดง error message ที่เข้าใจง่ายบนหน้าเว็บ
        # เมื่อ pipeline ล้มเหลว (เช่น หา JSON ไม่เจอ, Baseline ไม่ converge)
        # แทนที่จะโยน exception ดิบขึ้นไปให้ Flask จับ
        self.error: Optional[str] = None

        self.net: Optional[NetworkGraph]    = None
        self.json_path: Optional[Path]      = None
        self.raw: Optional[dict]            = None
        self.bus_to_node: Dict[str, int]    = {}
        self.snap_tol: float                = 2.0

        self.baseline: Optional[SimResult]  = None
        self.phase_raw: Optional[dict]      = None
        self.phase_result: Optional[SimResult] = None
        self.phase_moves: List[PhaseMove]   = []
        self.upgrade_options: List[UpgradeResult]  = []
        self.applied_upgrade: Optional[UpgradeResult] = None
        self.upgrade_result: Optional[SimResult] = None
        self.phase_moves_cu: List[PhaseMove] = []
        self.phase_result_cu: Optional[SimResult] = None
        self.phase_add_options: List[PhaseAddResult] = []
        self.applied_phase_add: Optional[PhaseAddResult] = None
        self.phase_moves2: List[PhaseMove]  = []        # phase transfer รอบ 2 (หลัง phase addition)
        self.phase_result2: Optional[SimResult] = None
        self.final_raw: Optional[dict]           = None  # final modified JSON after all steps
        self.final_result: Optional[SimResult]   = None  # combined result after all applied steps

        # ── Design check (มาตรฐานเฟสสาย) ───────────────────────────────
        self.design_comps: List[DesignComp]       = []   # ก่อนปรับปรุง
        self.design_summary: Dict[str, float]     = {}
        self.design_violations: List[DesignComp]  = []
        self.design_comps_after: List[DesignComp] = []   # หลังปรับปรุง
        self.design_summary_after: Dict[str, float] = {}
        self.design_violations_after: List[DesignComp] = []

    # ------------------------------------------------------------------

    def _ensure_json(self) -> Path:
        fac = self.facilityid
        candidates = [
            Path(self.json_dir) / f"NetworkLV{fac}_with_MV.json",
            Path(self.json_dir) / f"NetworkLV_{fac}_with_MV.json",
            Path(self.json_dir) / f"NetworkLV{fac}.json",
            Path(self.json_dir) / f"NetworkLV_{fac}.json",
        ]
        # ลบ JSON เดิมทุกไฟล์ก่อนดึงใหม่เสมอ
        for p in candidates:
            if p.exists():
                p.unlink()
                print(f"  [JSON] ลบ cache เดิม: {p.name}")
        print(f"  [JSON] ดึงข้อมูลจาก GIS API...")
        try:
            import InputJsonApi as inf
            inf.set_gis_region(self.region)
            result = inf.run_once_with_facilityid(fac, project_id=f"phaseopt_{fac}")
        except Exception as exc:
            raise FileNotFoundError(
                f"ไม่พบ JSON สำหรับ {fac} และ Stage 1 ล้มเหลว: {exc}") from exc
        out_path = result.get("out_path") if isinstance(result, dict) else None
        if out_path:
            p = Path(out_path)
            if not p.is_absolute():
                p = Path(self.json_dir) / p
            if p.exists():
                return p
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"ไม่พบ JSON สำหรับ {fac} แม้หลังรัน Stage 1")

    # ------------------------------------------------------------------

    def _has_problem(self, result: SimResult) -> bool:
        return (bool(result.low_v_nodes) or
                result.phase_imbalance_pct > self.max_imbalance_pct or
                result.max_phase_loading_pct > self.max_phase_loading_pct)

    # ── Design check — มาตรฐานเฟสสาย ───────────────────────────────────
    def _tx_mask(self) -> int:
        """เฟสที่หม้อแปลงมีจริง (bit mask) — 0 ถ้ายังไม่มี baseline"""
        if self.baseline is None or not self.baseline.converged:
            return 0
        mask = 0
        for pd in _tx_available_pds(self.baseline):
            mask |= pd
        return mask

    def _assess_design(self, raw: Optional[dict]) -> Tuple[List[DesignComp], Dict[str, float]]:
        """
        ตรวจว่าสาย LV เดินครบเฟสหม้อแปลงหรือไม่ — เกณฑ์เชิงมาตรฐานการออกแบบ
        ทำแม้ผลทางไฟฟ้าจะผ่านเกณฑ์แล้ว เพราะสายไม่ครบเฟสคือข้อจำกัดถาวรของระบบ
        (หม้อแปลง 1 เฟสข้าม — เพิ่มเฟสสายไม่ได้ ต้องเพิ่มเฟสหม้อแปลงก่อน)
        """
        if not self.design_check or self.net is None or raw is None:
            return [], {}
        mask = self._tx_mask()
        if bin(mask).count("1") < 2:
            return [], {}
        return assess_design(self.net, raw, mask,
                             self.design_min_len_m, self.design_min_meters)

    def _pa_ok(self, pa: PhaseAddResult, ref: SimResult, design_mode: bool) -> bool:
        """ข้อเสนอเพิ่มเฟสนี้ควรนำไปใช้หรือไม่ (เทียบกับสถานะ ref)"""
        if not pa.result.converged:
            return False
        if not _overload_ok(ref, pa.result, self.max_phase_loading_pct):
            return False                        # ห้ามทำให้เฟสหม้อแปลงเกินพิกัด
        if design_mode:
            # เหตุผลที่ทำคือ "สายต้องครบเฟส" — รับได้ถ้าไม่ทำให้แย่ลง
            return (pa.design_fixed > 0 and
                    pa.result.min_v >= self.min_voltage_v and
                    pa.delta_imbalance >= -3.0)
        if pa.delta_min_v > 0:
            return True
        d_ov = (_overload_excess(ref, self.max_phase_loading_pct) -
                _overload_excess(pa.result, self.max_phase_loading_pct))
        return (pa.delta_imbalance >= 3.0 or d_ov >= 3.0) and pa.delta_min_v >= -1.0

    @staticmethod
    def design_severity(ratio_full_pct: float) -> str:
        """ระดับความรุนแรงจากสัดส่วนความยาวสายที่ครบเฟสหม้อแปลง"""
        if ratio_full_pct < 50.0:
            return "เร่งด่วน"
        if ratio_full_pct < 70.0:
            return "ปานกลาง"
        return "ทำเมื่อมีโอกาส"

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"LV Phase / Conductor Optimizer")
        print(f"  FACILITYID   : {self.facilityid}")
        print(f"  min_voltage  : {self.min_voltage_v} V")
        print(f"  max_imbalance: {self.max_imbalance_pct}%")
        print(f"  max_phase_ld : {self.max_phase_loading_pct}% ของพิกัดเฟสหม้อแปลง (PHASEx_KVA)")
        print(f"{'='*60}")

        # ── Step 1: JSON ─────────────────────────────────────────────────
        print("\n[Step 1] Find TR & LV Network in JSON...")
        try:
            self.json_path = self._ensure_json()
        except FileNotFoundError as exc:
            print(f"  [!] {exc}")
            self.error = str(exc)
            return

        # ── Step 2: Parse network ─────────────────────────────────────────
        print("\n[Step 2] Parse network...")
        self.net = NetworkGraph(str(self.json_path))
        self.snap_tol = self.net.snap_tol
        print(f"  nodes={len(self.net.G.nodes())}  edges={len(self.net.G.edges())}  "
              f"kVA={self.net.rated_kva:.0f}  meters={len(self.net.load_feat_nodes)}")

        with open(self.json_path, encoding="utf-8") as fh:
            self.raw = json.load(fh)

        self.bus_to_node = _build_bus_to_node(self.net)

        # ── Step 3: Baseline simulation ───────────────────────────────────
        print("\n[Step 3] Baseline simulation...")
        self.baseline = _simulate(
            str(self.json_path), self.bus_to_node,
            self.net.rated_kva, self.min_voltage_v, self.snap_tol,
            phase_rating_kva=_tx_phase_rating_from_raw(self.raw))
        _print_sim_line("Baseline", self.baseline)

        if not self.baseline.converged:
            print(f"  [!] Baseline ไม่ converge: {self.baseline.error}")
            self.error = f"Baseline ไม่ converge: {self.baseline.error}"
            return

        # ── Step 3b: ตรวจมาตรฐานเฟสสาย (Design Check) ─────────────────────
        self.design_comps, self.design_summary = self._assess_design(self.raw)
        self.design_violations = [c for c in self.design_comps if c.violates]
        if self.design_summary:
            s = self.design_summary
            print(f"  [Design] สายครบเฟสหม้อแปลง {s['ratio_full_pct']:.0f}% ของความยาว  "
                  f"โหลดบนสายไม่ครบเฟส {s['kw_partial_pct']:.0f}%  "
                  f"กลุ่มไม่ครบเฟส {s['n_comps']} กลุ่ม "
                  f"(เข้าเกณฑ์ต้องเพิ่มเฟส {s['n_violate']} กลุ่ม = {s['len_violate']:.0f} m, "
                  f"{s['meters_violate']} มิเตอร์)")

        if not self._has_problem(self.baseline) and not self.design_violations:
            print("\n  [OK] ระบบปกติ — ไม่มีแรงดันตก, phase imbalance เกินเกณฑ์ "
                  "หรือสายไม่ครบเฟสตามหลักออกแบบ")
            # ไม่มีการแก้ไข → สถานะหลังปรับปรุง = สถานะเดิม (ให้รายงานมีค่าครบ)
            self.design_comps_after     = self.design_comps
            self.design_summary_after   = self.design_summary
            self.design_violations_after = []
            return

        if not self._has_problem(self.baseline):
            print(f"\n  [Design] ไฟฟ้าผ่านเกณฑ์ทุกข้อ แต่มีสายไม่ครบเฟสหม้อแปลง "
                  f"{len(self.design_violations)} กลุ่ม — ทำต่อตามหลักออกแบบ")

        # ── Step 3c: เพิ่มเฟสสายก่อน (เมื่อผิดหลักออกแบบ) ────────────────────
        # ทำก่อนย้ายมิเตอร์ตามที่ผู้ใช้กำหนด: สายที่ต้องเพิ่มเฟสก็ต้องเพิ่มอยู่ดี
        # เมื่อเพิ่มแล้วค่อยย้ายมิเตอร์ "ในกิ่งที่เพิ่งเพิ่มเฟส" จน %Unbalance เข้าเกณฑ์
        # → งานกระจุกที่เดียว ชุดปฏิบัติงานไปครั้งเดียวจบ และเฟสใหม่มีโหลดใช้งานจริง
        pt_raw, pt_ref = self.raw, self.baseline
        if self.design_violations:
            print(f"\n[Step 3c] เพิ่มเฟสสายตามหลักออกแบบก่อนย้ายมิเตอร์ "
                  f"({len(self.design_violations)} กลุ่ม)...")
            self.phase_add_options = simulate_phase_addition(
                self.net, self.raw, self.baseline,
                self.bus_to_node, self.min_voltage_v, self.snap_tol,
                max_imbalance_pct=self.max_imbalance_pct,
                max_phase_loading_pct=self.max_phase_loading_pct,
                design_min_len_m=self.design_min_len_m,
                design_min_meters=self.design_min_meters,
                design_check=self.design_check,
                prefer_design=True)
            if self.phase_add_options and self._pa_ok(
                    self.phase_add_options[0], self.baseline, design_mode=True):
                best_pa = self.phase_add_options[0]
                self.applied_phase_add = best_pa
                print(f"  ใช้: {len(best_pa.upgraded_edges)} edges→"
                      f"{_PD_LABEL.get(best_pa.to_pd, 'ABC')}  "
                      f"ย้าย {len(best_pa.meter_moves)} มิเตอร์ในกิ่งที่เพิ่มเฟส")
                pa_raw = copy.deepcopy(self.raw)
                for fi, _eu, _ev in best_pa.upgraded_edges:
                    pa_raw["features"][fi].setdefault(
                        "attributes", {})["PHASEDESIGNATION"] = best_pa.to_pd
                for fi, new_m_pd in best_pa.meter_moves:
                    _apply_meter_phase(pa_raw["features"], fi, new_m_pd)
                _print_sim_line("After Phase Addition", best_pa.result)
                pt_raw, pt_ref = pa_raw, best_pa.result
                self.final_raw, self.final_result = pa_raw, best_pa.result

        pa_applied_early = self.applied_phase_add is not None

        # ── Step 4: Phase transfer ─────────────────────────────────────────
        print("\n[Step 4] ย้ายเฟสมิเตอร์ (Phase Transfer)...")
        self.phase_raw, self.phase_moves = optimize_phase_transfer(
            self.net, pt_raw, pt_ref, self.bus_to_node,
            self.min_voltage_v, snap_tol=self.snap_tol,
            max_imbalance_pct=self.max_imbalance_pct, time_limit_s=self.max_sim_time_s,
            min_gain_kw=self.min_move_gain_kw,
            max_phase_loading_pct=self.max_phase_loading_pct)

        # Re-simulate combined result after all moves
        if self.phase_moves:
            self.phase_result = _sim_modified_json(
                self.phase_raw, self.bus_to_node,
                self.net.rated_kva, self.min_voltage_v, self.snap_tol)
            _print_sim_line("After Phase Transfer", self.phase_result)
        else:
            self.phase_result = pt_ref

        # Source for further analysis: use post-phase-transfer state
        analysis_raw    = self.phase_raw or pt_raw
        analysis_result = self.phase_result
        self.final_raw    = analysis_raw
        self.final_result = analysis_result

        # ── Step 5: Conductor upgrade (if still problems) ─────────────────
        if self._has_problem(analysis_result):
            print("\n[Step 5] เพิ่มขนาดสาย (Conductor Upgrade)...")
            self.upgrade_options = simulate_conductor_upgrade(
                self.net, analysis_raw, analysis_result,
                self.bus_to_node, self.min_voltage_v, self.snap_tol)

        # ── Step 5b: Apply best conductor upgrade if it helps ─────────────
        if (self.upgrade_options and self.upgrade_options[0].delta_min_v > 0
                and _overload_ok(analysis_result, self.upgrade_options[0].result,
                                 self.max_phase_loading_pct)):
            best_up = self.upgrade_options[0]
            self.applied_upgrade = best_up
            u_up, v_up = best_up.edge
            print(f"\n[Step 5b] ใช้ Conductor Upgrade: subtree ({u_up}→{v_up})  "
                  f"{len(best_up.upgraded_edges)} segment →{best_up.to_size}mm²  "
                  f"{best_up.n_affected} โหนด LowV")
            upgrade_raw = copy.deepcopy(analysis_raw)
            for _fi, _, _ in best_up.upgraded_edges:
                upgrade_raw["features"][_fi].setdefault(
                    "attributes", {})["CONDUCTORSIZE"] = best_up.to_size

            self.upgrade_result = _sim_modified_json(
                upgrade_raw, self.bus_to_node,
                self.net.rated_kva, self.min_voltage_v, self.snap_tol)
            _print_sim_line("After Conductor Upgrade", self.upgrade_result)

            # ── Step 5c: Phase Transfer หลัง Conductor Upgrade ───────────
            print("\n[Step 5c] Phase Transfer หลัง Conductor Upgrade...")
            upgrade_raw2, self.phase_moves_cu = optimize_phase_transfer(
                self.net, upgrade_raw, self.upgrade_result, self.bus_to_node,
                self.min_voltage_v, snap_tol=self.snap_tol,
                max_imbalance_pct=self.max_imbalance_pct, time_limit_s=self.max_sim_time_s,
            min_gain_kw=self.min_move_gain_kw,
            max_phase_loading_pct=self.max_phase_loading_pct)
            if self.phase_moves_cu:
                self.phase_result_cu = _sim_modified_json(
                    upgrade_raw2, self.bus_to_node,
                    self.net.rated_kva, self.min_voltage_v, self.snap_tol)
                _print_sim_line("After PT (post-Upgrade)", self.phase_result_cu)
                analysis_raw    = upgrade_raw2
                analysis_result = self.phase_result_cu
            else:
                self.phase_result_cu = self.upgrade_result
                analysis_raw    = upgrade_raw
                analysis_result = self.upgrade_result
            self.final_raw = analysis_raw

        # ── Step 6: Phase addition (if still problems) ────────────────────
        design_pending = [c for c in self._assess_design(analysis_raw)[0] if c.violates]
        if (self._has_problem(analysis_result) or design_pending) and not self.applied_phase_add:
            if not self._has_problem(analysis_result):
                print(f"\n[Step 6] เพิ่มเฟสสายตามหลักออกแบบ "
                      f"({len(design_pending)} กลุ่มไม่ครบเฟสหม้อแปลง)...")
            else:
                print("\n[Step 6] เพิ่มเฟสสาย (Phase Addition)...")
            self.phase_add_options = simulate_phase_addition(
                self.net, analysis_raw, analysis_result,
                self.bus_to_node, self.min_voltage_v, self.snap_tol,
                max_imbalance_pct=self.max_imbalance_pct,
                max_phase_loading_pct=self.max_phase_loading_pct,
                design_min_len_m=self.design_min_len_m,
                design_min_meters=self.design_min_meters,
                design_check=self.design_check)

        # ── Step 7: Apply best phase addition if it actually improves ──────
        # ยอมรับถ้า Vmin ดีขึ้น  หรือ  imbalance ลดลง ≥3% โดย Vmin ไม่แย่ลงเกิน 1V
        def _pa_worth_applying(pa: PhaseAddResult) -> bool:
            if not pa.result.converged:
                return False
            if not _overload_ok(analysis_result, pa.result, self.max_phase_loading_pct):
                return False                    # ห้ามทำให้เฟสหม้อแปลงเกินพิกัด
            # โหมดมาตรฐานออกแบบ: ไฟฟ้าผ่านเกณฑ์อยู่แล้ว เหตุผลที่ทำคือ "สายต้องครบเฟส"
            # จึงรับได้ตราบใดที่ไม่ทำให้แย่ลง (ไม่สร้างแรงดันต่ำ / imbalance ไม่พุ่ง)
            if not self._has_problem(analysis_result) and design_pending:
                return (pa.design_fixed > 0 and
                        pa.result.min_v >= self.min_voltage_v and
                        pa.delta_imbalance >= -3.0)
            if pa.delta_min_v > 0:
                return True
            d_ov = (_overload_excess(analysis_result, self.max_phase_loading_pct) -
                    _overload_excess(pa.result, self.max_phase_loading_pct))
            return (pa.delta_imbalance >= 3.0 or d_ov >= 3.0) and pa.delta_min_v >= -1.0

        if (self.phase_add_options and not pa_applied_early
                and _pa_worth_applying(self.phase_add_options[0])):
            best_pa = self.phase_add_options[0]
            self.applied_phase_add = best_pa
            n_up = len(best_pa.upgraded_edges)
            print(f"\n[Step 7] ใช้ Phase Addition: {n_up} edges→{_PD_LABEL.get(best_pa.to_pd, 'ABC')}  "
                  f"ย้าย {len(best_pa.meter_moves)} มิเตอร์")
            _print_sim_line("After Phase Addition", best_pa.result)

            # ── Step 8: Phase Transfer รอบ 2 หลัง Phase Addition ──────────
            print("\n[Step 8] Phase Transfer รอบ 2 หลัง Phase Addition...")
            combined_raw = copy.deepcopy(analysis_raw)
            for fi, eu, ev in best_pa.upgraded_edges:
                combined_raw["features"][fi].setdefault("attributes", {})["PHASEDESIGNATION"] = best_pa.to_pd
            for fi, new_m_pd in best_pa.meter_moves:
                _apply_meter_phase(combined_raw["features"], fi, new_m_pd)

            combined_raw2, self.phase_moves2 = optimize_phase_transfer(
                self.net, combined_raw, best_pa.result, self.bus_to_node,
                self.min_voltage_v, snap_tol=self.snap_tol,
                max_imbalance_pct=self.max_imbalance_pct, time_limit_s=self.max_sim_time_s,
            min_gain_kw=self.min_move_gain_kw,
            max_phase_loading_pct=self.max_phase_loading_pct)

            if self.phase_moves2:
                self.phase_result2 = _sim_modified_json(
                    combined_raw2, self.bus_to_node,
                    self.net.rated_kva, self.min_voltage_v, self.snap_tol)
                _print_sim_line("Combined Final", self.phase_result2)
                self.final_result = self.phase_result2
                self.final_raw    = combined_raw2
            else:
                self.phase_result2 = best_pa.result
                self.final_result  = best_pa.result
                self.final_raw     = combined_raw
                print("  ไม่มีการย้ายเพิ่มเติม")
        else:
            if self.applied_upgrade:
                self.final_result = (self.phase_result_cu
                                     if self.phase_result_cu else self.upgrade_result)
            else:
                self.final_result = self.phase_result if self.phase_result else self.baseline

        # ── ตรวจมาตรฐานเฟสสายซ้ำบนผลสุดท้าย (หลังเพิ่มเฟสแล้วเหลือกี่กลุ่ม) ──
        self.design_comps_after, self.design_summary_after = self._assess_design(
            self.final_raw if self.final_raw is not None else analysis_raw)
        self.design_violations_after = [c for c in self.design_comps_after if c.violates]
        if self.design_summary_after:
            s = self.design_summary_after
            print(f"\n[Design] ผลสุดท้าย: สายครบเฟสหม้อแปลง {s['ratio_full_pct']:.0f}%  "
                  f"เหลือกลุ่มที่ยังไม่ครบเฟสและเข้าเกณฑ์ {s['n_violate']} กลุ่ม")


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_sim_line(label: str, r: SimResult) -> None:
    if not r.converged:
        print(f"  [{label}] ไม่ converge: {r.error}")
        return
    phase_str = "  ".join(
        f"P{PHASE_MAP.get(n,'?')}={kw:.1f}kW"
        + (f"({r.phase_loading_pct[n]:.0f}%)" if n in r.phase_loading_pct else "")
        for n, kw in sorted(r.phase_kw.items()))
    print(f"  [{label}]  Vmin={r.min_v:.1f}V  TR={r.tr_loading_pct:.1f}%  "
          f"PhMax={r.max_phase_loading_pct:.0f}%  "
          f"Imbal={r.phase_imbalance_pct:.1f}%  LowV={len(r.low_v_nodes)}nodes"
          + (f"  [{phase_str}]" if phase_str else ""))


def print_console_report(opt: LVOptimizer) -> None:
    col = 72
    net = opt.net
    b   = opt.baseline
    print(f"\n{'='*col}")
    print(f"  PHASE OPTIMIZER — {net.facilityid}  "
          f"(rated={net.rated_kva:.0f} kVA  meters={len(net.load_feat_nodes)})")
    print(f"{'='*col}")

    print("\nBaseline:")
    _print_sim_line("", b)
    if b.phase_kw:
        for node, kw in sorted(b.phase_kw.items()):
            print(f"  Phase {PHASE_MAP.get(node,'?')}: {kw:.2f} kW")

    if opt.phase_moves:
        pr = opt.phase_result
        n_bu = sum(1 for m in opt.phase_moves if m.stage == "bottomup")
        print(f"\nPhase Transfer ({len(opt.phase_moves)} moves — "
              f"branch→main {n_bu}, greedy {len(opt.phase_moves)-n_bu}):")
        for m in sorted(opt.phase_moves,
                        key=lambda x: (0 if x.stage == "bottomup" else 1, -x.level)):
            src = f"[{m.scope}]" if m.stage == "bottomup" else "[greedy]"
            print(f"  {m.meter.peano or 'idx='+str(m.meter.feat_idx):>14}  "
                  f"{m.meter.kw:6.2f}kW  "
                  f"{PD_TO_PHASE.get(m.from_pd,'?')}→{PD_TO_PHASE.get(m.to_pd,'?')}  "
                  f"{src}")
        if pr:
            print(f"  ผลรวม: Vmin={pr.min_v:.1f}V  "
                  f"Imbal={pr.phase_imbalance_pct:.1f}%  LowV={len(pr.low_v_nodes)}")
    else:
        print("\nPhase Transfer: ไม่มีการย้าย")

    if opt.upgrade_options:
        print(f"\nConductor Upgrade (top {min(3, len(opt.upgrade_options))}):")
        for i, up in enumerate(opt.upgrade_options[:3], 1):
            status = (f"Vmin={up.result.min_v:.1f}V  ΔV={up.delta_min_v:+.1f}V"
                      if up.result.converged else "[NG]")
            marker = " <- APPLIED" if (up is opt.applied_upgrade) else ""
            print(f"  #{i} subtree ({up.edge[0]}→{up.edge[1]})  "
                  f"{len(up.upgraded_edges)} seg →{up.to_size}mm²  "
                  f"{up.n_affected}LowV  {status}{marker}")
            if up.ev_peanos:
                print(f"       EV: {', '.join(up.ev_peanos)}")
    else:
        print("\nConductor Upgrade: ไม่มีตัวเลือก")

    if opt.applied_upgrade and opt.upgrade_result:
        ur = opt.upgrade_result
        print(f"\nPhase Transfer หลัง CU ({len(opt.phase_moves_cu)} moves):")
        for m in opt.phase_moves_cu:
            print(f"  {m.meter.peano or 'idx='+str(m.meter.feat_idx):>14}  "
                  f"{PD_TO_PHASE.get(m.from_pd,'?')}→{PD_TO_PHASE.get(m.to_pd,'?')}  "
                  f"ΔV={m.delta_min_v:+.1f}V  Δimbal={m.delta_imbalance:+.1f}%")
        if not opt.phase_moves_cu:
            print("  ไม่มีการย้าย")
        if opt.phase_result_cu:
            pur = opt.phase_result_cu
            print(f"  ผลรวม: Vmin={pur.min_v:.1f}V  "
                  f"Imbal={pur.phase_imbalance_pct:.1f}%  LowV={len(pur.low_v_nodes)}")

    if opt.phase_add_options:
        print(f"\nPhase Addition (top {min(3, len(opt.phase_add_options))}):")
        for i, pa in enumerate(opt.phase_add_options[:3], 1):
            status = (f"Vmin={pa.result.min_v:.1f}V  ΔV={pa.delta_min_v:+.1f}V  "
                      f"Δimbal={pa.delta_imbalance:+.1f}%"
                      if pa.result.converged else "[NG]")
            marker = " <- APPLIED" if (pa is opt.applied_phase_add) else ""
            n_up = len(pa.upgraded_edges) if pa.upgraded_edges else 1
            print(f"  #{i} component ({n_up} edges)  "
                  f"->{_PD_LABEL.get(pa.to_pd, 'ABC')}  {len(pa.meter_moves)}meters  {status}{marker}")
    else:
        print("\nPhase Addition: ไม่มีตัวเลือก")

    # ── Design Check — มาตรฐานเฟสสาย ────────────────────────────────────
    if opt.design_summary:
        s = opt.design_summary
        print(f"\nDesign Check — มาตรฐานเฟสสาย (เทียบกับเฟสที่หม้อแปลงมี):")
        print(f"  สายครบเฟสหม้อแปลง {s['ratio_full_pct']:.1f}% ของความยาว "
              f"({s['len_total']-s['len_partial']:.0f}/{s['len_total']:.0f} m)  "
              f"ระดับ: {LVOptimizer.design_severity(s['ratio_full_pct'])}")
        print(f"  โหลดบนสายไม่ครบเฟส {s['kw_partial']:.1f}/{s['kw_total']:.1f} kW "
              f"({s['kw_partial_pct']:.0f}%)")
        for c in opt.design_comps:
            mark = "ต้องเพิ่มเฟส" if c.violates else "ซอยสั้น — ไม่บังคับ"
            print(f"    {c.label}  [{mark}]")
        n_after = len(opt.design_violations_after)
        if opt.design_violations:
            fixed = len(opt.design_violations) - n_after
            print(f"  สรุป: เข้าเกณฑ์ต้องเพิ่มเฟส {len(opt.design_violations)} กลุ่ม "
                  f"({s['len_violate']:.0f} m, {s['meters_violate']} มิเตอร์) — "
                  f"แก้ในรอบนี้ {fixed} กลุ่ม, เหลือ {n_after} กลุ่ม")
        else:
            print("  สรุป: สายทุกกลุ่มครบเฟสหม้อแปลง หรือเป็นซอยสั้นที่ไม่บังคับ ✓")

    if opt.phase_moves2:
        pr2 = opt.phase_result2
        print(f"\nPhase Transfer รอบ 2 ({len(opt.phase_moves2)} moves หลัง Phase Addition):")
        for m in opt.phase_moves2:
            print(f"  {m.meter.peano or 'idx='+str(m.meter.feat_idx):>14}  "
                  f"{PD_TO_PHASE.get(m.from_pd,'?')}→{PD_TO_PHASE.get(m.to_pd,'?')}  "
                  f"ΔV={m.delta_min_v:+.1f}V  Δimbal={m.delta_imbalance:+.1f}%")
        if pr2:
            print(f"  ผลรวม: Vmin={pr2.min_v:.1f}V  "
                  f"Imbal={pr2.phase_imbalance_pct:.1f}%  LowV={len(pr2.low_v_nodes)}")

    if (opt.applied_phase_add or opt.applied_upgrade) and opt.final_result:
        fr = opt.final_result
        _steps = []
        if opt.phase_moves:       _steps.append("PT1")
        if opt.applied_upgrade:
            _cu = opt.applied_upgrade
            _steps.append(f"CU({len(_cu.upgraded_edges)}seg→{_cu.to_size}mm²)")
        if opt.phase_moves_cu:    _steps.append("PT_CU")
        if opt.applied_phase_add:
            _pa = opt.applied_phase_add
            _steps.append(f"PA({len(_pa.upgraded_edges)}edges→{_PD_LABEL.get(_pa.to_pd, 'ABC')})")
        if opt.phase_moves2:      _steps.append("PT2")
        steps = " + ".join(_steps) if _steps else "—"
        print(f"\n{'─'*col}")
        print(f"  COMBINED FINAL  ({steps})")
        print(f"{'─'*col}")
        _print_sim_line("", fr)
        if fr.phase_kw:
            for node, kw in sorted(fr.phase_kw.items()):
                print(f"  Phase {PHASE_MAP.get(node,'?')}: {kw:.2f} kW")
        problems = []
        if fr.low_v_nodes:
            problems.append(f"LowV={len(fr.low_v_nodes)} โหนด")
        if fr.phase_imbalance_pct > opt.max_imbalance_pct:
            problems.append(f"Imbal={fr.phase_imbalance_pct:.1f}%>{opt.max_imbalance_pct}%")
        if fr.max_phase_loading_pct > opt.max_phase_loading_pct:
            problems.append(f"เฟสหม้อแปลงโหลด {fr.max_phase_loading_pct:.0f}%"
                            f">{opt.max_phase_loading_pct:.0f}%")
        if opt.design_violations_after:
            problems.append(f"สายไม่ครบเฟสหม้อแปลง {len(opt.design_violations_after)} กลุ่ม")
        print(f"  สถานะ: {'ผ่านเกณฑ์ ✓' if not problems else 'ยังเกินเกณฑ์: ' + ', '.join(problems)}")

    # ── พิกัดหม้อแปลงรายเฟส (ผลสุดท้าย) ───────────────────────────────
    _fr = opt.final_result or opt.phase_result or opt.baseline
    if _fr is not None and _fr.converged and _fr.phase_loading_pct:
        print()
        print("  พิกัดหม้อแปลงรายเฟส (ผลสุดท้าย): " + "  ".join(
            f"{PHASE_MAP.get(n, '?')}={_fr.phase_kva.get(n, 0):.1f}/"
            f"{_fr.phase_rating_kva.get(n, 0):.1f}kVA({v:.0f}%)"
            for n, v in sorted(_fr.phase_loading_pct.items())))
        for _msg in _tr_rating_advice(_fr, opt.max_phase_loading_pct):
            print(f"  [!] {_msg}")

    print(f"\n{'='*col}")


# ─────────────────────────────────────────────────────────────────────────────
# Excel report
# ─────────────────────────────────────────────────────────────────────────────

def save_excel_report(opt: LVOptimizer, out_path: str) -> None:
    wb = openpyxl.Workbook()
    HDR  = PatternFill("solid", fgColor="1F497D")
    HFNT = Font(color="FFFFFF", bold=True)
    OK   = PatternFill("solid", fgColor="E2EFDA")
    NG   = PatternFill("solid", fgColor="FCE4D6")
    WARN = PatternFill("solid", fgColor="FFEB9C")

    def _hdr(ws, cols):
        ws.append(cols)
        for c in ws[1]:
            c.fill = HDR; c.font = HFNT

    def _autowidth(ws):
        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 36)

    net = opt.net
    b   = opt.baseline
    pr  = opt.phase_result

    # ── Summary ──────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    rows = [
        ["FACILITYID",        net.facilityid],
        ["Transformer kVA",   net.rated_kva],
        ["Network nodes",     len(net.G.nodes())],
        ["Network edges",     len(net.G.edges())],
        ["Total meters",      len(net.load_feat_nodes)],
        [""],
        ["=== Baseline ===", ""],
        ["Vmin (V)",          round(b.min_v, 1)         if b else ""],
        ["TR loading (%)",    round(b.tr_loading_pct, 1) if b else ""],
        ["Max phase loading (% พิกัดเฟส)", round(b.max_phase_loading_pct, 1) if b else ""],
        ["Phase imbalance (%)".replace("%","%%"), round(b.phase_imbalance_pct, 1) if b else ""],
        ["Low-voltage nodes", len(b.low_v_nodes)         if b else ""],
    ]
    if b and b.phase_kw:
        rows.append([""])
        for node, kw in sorted(b.phase_kw.items()):
            rows.append([f"Phase {PHASE_MAP.get(node,'?')} (kW)", round(kw, 2)])
        for node in sorted(b.phase_loading_pct):
            rows.append([f"Loading {PHASE_MAP.get(node,'?')} (% ของ "
                         f"{b.phase_rating_kva.get(node, 0):.1f} kVA)",
                         round(b.phase_loading_pct[node], 1)])

    if opt.phase_moves and pr:
        rows += [
            [""],
            ["=== After Phase Transfer ===", ""],
            ["Vmin (V)",             round(pr.min_v, 1)],
            ["Phase imbalance (%%)". replace("%%","%"), round(pr.phase_imbalance_pct, 1)],
            ["Low-voltage nodes",    len(pr.low_v_nodes)],
            ["Meters moved",         len(opt.phase_moves)],
        ]

    if opt.upgrade_options:
        best = opt.upgrade_options[0]
        _cu_label = "APPLIED" if (opt.applied_upgrade and opt.applied_upgrade is best) else "ตัวเลือก"
        rows += [
            [""],
            [f"=== Conductor Upgrade ({_cu_label}) ===", ""],
            ["Entry edge",          f"{best.edge[0]} → {best.edge[1]}"],
            ["Conductor upgrade",   f"→ {best.to_size} mm²  ({len(best.upgraded_edges)} segment บน path)"],
            ["Affected LowV nodes", best.n_affected],
            ["EV meters (เหตุที่เพิ่มขนาดสาย)", ", ".join(best.ev_peanos)],
            ["ΔVmin sim (V)",       round(best.delta_min_v, 1)],
            ["Sim Vmin (V)",        round(best.result.min_v, 1) if best.result.converged else "NG"],
        ]
        if opt.applied_upgrade and opt.upgrade_result and opt.upgrade_result.converged:
            rows += [
                ["After Apply Vmin (V)",  round(opt.upgrade_result.min_v, 1)],
                ["After Apply Imbal (%)", round(opt.upgrade_result.phase_imbalance_pct, 1)],
                ["PT หลัง CU (moves)",   len(opt.phase_moves_cu)],
            ]
            if opt.phase_result_cu and opt.phase_result_cu.converged:
                rows.append(["After CU+PT Vmin (V)", round(opt.phase_result_cu.min_v, 1)])

    if opt.phase_add_options:
        best = opt.phase_add_options[0]
        _pa_label = "APPLIED" if (opt.applied_phase_add is best) else "ตัวเลือก"
        rows += [
            [""],
            [f"=== Best Phase Addition ({_pa_label}) ===", ""],
            ["Edge",                    f"{best.edge[0]} → {best.edge[1]}"],
            ["Phase upgrade",           f"{_PD_LABEL.get(best.from_pd,'?')} → ABC"],
            ["Edges upgraded",          len(best.upgraded_edges)],
            ["Meters redistributed",    len(best.meter_moves)],
            ["Affected LowV nodes",     best.n_affected],
            ["ΔVmin (V)",               round(best.delta_min_v, 1)],
            ["Δimbalance (%)",          round(best.delta_imbalance, 1)],
            ["After Vmin (V)",   round(best.result.min_v, 1) if best.result.converged else "NG"],
            ["After Imbal (%)",  round(best.result.phase_imbalance_pct, 1) if best.result.converged else "NG"],
        ]

    if opt.design_summary:
        _ds = opt.design_summary
        rows += [
            [""],
            ["=== Design Check — มาตรฐานเฟสสาย ===", ""],
            ["สายครบเฟสหม้อแปลง (% ความยาว)", round(_ds["ratio_full_pct"], 1)],
            ["ความยาวสาย LV รวม (m)",          round(_ds["len_total"], 0)],
            ["ความยาวสายไม่ครบเฟส (m)",        round(_ds["len_partial"], 0)],
            ["โหลดบนสายไม่ครบเฟส (%)",         round(_ds["kw_partial_pct"], 1)],
            ["ระดับความรุนแรง",                LVOptimizer.design_severity(_ds["ratio_full_pct"])],
            ["กลุ่มสายไม่ครบเฟส (กลุ่ม)",       _ds["n_comps"]],
            [f"เข้าเกณฑ์ต้องเพิ่มเฟส (≥{opt.design_min_len_m:.0f}m หรือ "
             f"≥{opt.design_min_meters} มิเตอร์)", _ds["n_violate"]],
            ["ความยาวที่ต้องเดินเพิ่มเฟส (m)",  round(_ds["len_violate"], 0)],
            ["เหลือกลุ่มที่ยังไม่ครบเฟส (หลังปรับปรุง)", len(opt.design_violations_after)],
        ]

    _fr = opt.final_result or opt.phase_result or b
    if _fr is not None and _fr.converged and _fr.phase_loading_pct:
        rows += [[""], ["=== พิกัดหม้อแปลงรายเฟส (ผลสุดท้าย) ===", ""]]
        for node in sorted(_fr.phase_loading_pct):
            rows.append([f"Loading {PHASE_MAP.get(node,'?')} final (%)",
                         round(_fr.phase_loading_pct[node], 1)])
        for _msg in (_tr_rating_advice(_fr, opt.max_phase_loading_pct)
                     or ["ไม่เกินพิกัดรายเฟส"]):
            rows.append(["คำแนะนำหม้อแปลง", _msg])

    for row in rows:
        ws.append(row)
    _autowidth(ws)

    # ── Phase Transfer ────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Phase Transfer")
    _hdr(ws2, ["#", "PEANO", "NodeID", "Load (kW)", "From Phase", "To Phase",
               "ขั้นตอน", "ระดับ (ลึก)", "กลุ่มที่จัดสมดุล", "ΔVmin (V)", "Δimbal (%)"])
    # เรียงจาก branch ลึกสุด → ไลน์เมน ให้ตรงกับลำดับที่ช่างจะไล่ทำหน้างาน
    _ordered = sorted(opt.phase_moves,
                      key=lambda m: (0 if m.stage == "bottomup" else 1, -m.level))
    for i, m in enumerate(_ordered, 1):
        ws2.append([
            i,
            m.meter.peano or f"idx={m.meter.feat_idx}",
            m.meter.node_id,
            round(m.meter.kw, 3),
            PD_TO_PHASE.get(m.from_pd, str(m.from_pd)),
            PD_TO_PHASE.get(m.to_pd,   str(m.to_pd)),
            "Branch→Main" if m.stage == "bottomup" else "Greedy (แรงดัน)",
            m.level if m.level >= 0 else "",
            m.scope,
            round(m.delta_min_v, 2),
            round(m.delta_imbalance, 2),
        ])
        fill = OK if (m.stage == "bottomup" or m.delta_min_v >= 0) else NG
        for c in ws2[ws2.max_row]: c.fill = fill
    _autowidth(ws2)

    # ── Conductor Upgrade ─────────────────────────────────────────────────
    ws3 = wb.create_sheet("Conductor Upgrade")
    _hdr(ws3, ["#", "Entry edge (u→v)", "Segments upgraded", "From mm²", "To mm²",
               "LowV nodes affected", "ΔVmin (V)", "After Vmin (V)", "Converged",
               "EV meters (เหตุที่เพิ่มขนาดสาย)"])
    for i, up in enumerate(opt.upgrade_options, 1):
        _from_str = "/".join(str(s) for s in sorted(set(up.from_sizes.values()))) or str(up.from_size)
        ws3.append([
            i, f"{up.edge[0]}→{up.edge[1]}",
            len(up.upgraded_edges), _from_str, up.to_size, up.n_affected,
            round(up.delta_min_v, 1),
            round(up.result.min_v, 1) if up.result.converged else "",
            "YES" if up.result.converged else "NO",
            ", ".join(up.ev_peanos),
        ])
        fill = OK if up.delta_min_v > 0 else NG
        for c in ws3[ws3.max_row]: c.fill = fill
    _autowidth(ws3)

    # ── Conductor Upgrade — รายละเอียด segment ────────────────────────────
    ws3b = wb.create_sheet("CU Segments")
    _hdr(ws3b, ["Option #", "Entry edge", "FeatIdx", "u", "v", "From mm²", "To mm²", "APPLIED?"])
    for i, up in enumerate(opt.upgrade_options, 1):
        applied = (up is opt.applied_upgrade)
        for fi, eu, ev in up.upgraded_edges:
            ws3b.append([
                i, f"{up.edge[0]}→{up.edge[1]}", fi, eu, ev,
                up.from_sizes.get(fi, ""), up.to_size,
                "YES" if applied else "",
            ])
            if applied:
                for c in ws3b[ws3b.max_row]: c.fill = OK
    _autowidth(ws3b)

    # ── Phase Addition ────────────────────────────────────────────────────
    ws4 = wb.create_sheet("Phase Addition")
    _hdr(ws4, ["#", "Edge (u→v)", "From phases", "To phases", "Edges upgraded",
               "Meters moved", "LowV nodes affected",
               "ΔVmin (V)", "Δimbal (%)", "After Vmin (V)", "After Imbal (%)",
               "Converged", "APPLIED?"])
    for i, pa in enumerate(opt.phase_add_options, 1):
        applied = (pa is opt.applied_phase_add)
        ws4.append([
            i, f"{pa.edge[0]}→{pa.edge[1]}",
            _PD_LABEL.get(pa.from_pd, str(pa.from_pd)), _PD_LABEL.get(pa.to_pd, "ABC"),
            len(pa.upgraded_edges),
            len(pa.meter_moves), pa.n_affected,
            round(pa.delta_min_v, 1), round(pa.delta_imbalance, 1),
            round(pa.result.min_v, 1) if pa.result.converged else "",
            round(pa.result.phase_imbalance_pct, 1) if pa.result.converged else "",
            "YES" if pa.result.converged else "NO",
            "YES" if applied else "",
        ])
        fill = OK if (pa.delta_min_v > 0 or pa.delta_imbalance > 0) else NG
        for c in ws4[ws4.max_row]: c.fill = fill
    _autowidth(ws4)

    # ── Design Check — รายกลุ่มสายที่ไม่ครบเฟสหม้อแปลง ─────────────────────
    ws4b = wb.create_sheet("Design Check")
    _hdr(ws4b, ["กลุ่ม", "ความยาว (m)", "Segments", "มิเตอร์", "Load (kW)",
                "เฟสปัจจุบัน", "ต้องเพิ่มเฟส?", "สถานะหลังปรับปรุง"])
    _after_nodes = [c.nodes for c in opt.design_violations_after]
    for c in opt.design_comps:
        still = any(c.nodes & n for n in _after_nodes)
        state = ("ยังไม่ครบเฟส" if still else
                 ("เพิ่มเฟสแล้ว" if c.violates else "ไม่บังคับ"))
        ws4b.append([
            c.idx + 1, round(c.length_m, 0), len(c.edges), c.n_meters, round(c.kw, 2),
            "/".join(_PD_LABEL.get(p, str(p)) for p in sorted(c.pds)),
            "ใช่" if c.violates else "ไม่ (ซอยสั้น)",
            state,
        ])
        if c.violates:
            ws4b.cell(ws4b.max_row, 7).fill = NG if still else OK
    _autowidth(ws4b)

    # ── Meter Detail (all meters) ─────────────────────────────────────────
    ws5 = wb.create_sheet("Meters")
    _hdr(ws5, ["FeatIdx", "PEANO", "NodeID", "Phase final (PD)", "Conductor PD",
               "Conductor phases", "Load kW", "Movable?"])
    _raw_for_meters = opt.final_raw if opt.final_raw is not None else opt.raw
    for m in _build_meter_inventory(opt.net, _raw_for_meters):
        ws5.append([
            m.feat_idx, m.peano, m.node_id,
            PD_TO_PHASE.get(m.current_pd, str(m.current_pd)),
            m.conductor_pd, _PD_LABEL.get(m.conductor_pd, str(m.conductor_pd)),
            round(m.kw, 3),
            "YES" if (m.is_single_phase and m.movable_pds) else "NO",
        ])
    _autowidth(ws5)

    wb.save(out_path)
    print(f"[Excel] บันทึก: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Static PNG map
# ─────────────────────────────────────────────────────────────────────────────

def draw_map(opt: LVOptimizer, out_path: str) -> None:
    net  = opt.net
    b    = opt.baseline
    # ใช้ผลลัพธ์สุดท้าย (final_result) แสดงแรงดัน — ถ้าไม่มีใช้ baseline
    display_r = opt.final_result if opt.final_result else b
    node_v = {nid: min(ph.values()) for nid, ph in display_r.node_phase_v.items() if ph}
    has_v  = bool(node_v)

    norm  = mcolors.Normalize(vmin=150, vmax=240)
    cmap  = mcolors.LinearSegmentedColormap.from_list(
        "volt", ["#d73027", "#fee08b", "#91cf60", "#1a9850"], N=256)
    BASE  = "#4472C4"

    fig, ax = plt.subplots(figsize=(15, 10))

    # Edges
    for u, v in net.G.edges():
        xu, yu = net.node_coords[u]; xv, yv = net.node_coords[v]
        ax.plot([xu, xv], [yu, yv], color=BASE, lw=1.2, alpha=0.45, zorder=2)

    # Nodes coloured by final voltage
    for nid, (x, y) in net.node_coords.items():
        s = 22 + net.node_kw.get(nid, 0) * 1.4
        v = node_v.get(nid)
        c = cmap(norm(v)) if v is not None else BASE
        ax.scatter(x, y, color=c, s=s, zorder=3, alpha=0.85)

    # Red ring — โหนดที่ยังมีแรงดันต่ำหลัง optimize (final state)
    for nid in display_r.low_v_nodes:
        if nid in net.node_coords:
            x, y = net.node_coords[nid]
            ax.scatter(x, y, s=75, facecolors="none", edgecolors="red",
                       linewidths=1.5, zorder=4)

    # Green ring for meters that were moved (all PT rounds)
    moved_nids = (
        {m.meter.node_id for m in opt.phase_moves} |
        {m.meter.node_id for m in opt.phase_moves_cu} |
        {m.meter.node_id for m in opt.phase_moves2}
    )
    for nid in moved_nids:
        if nid in net.node_coords:
            x, y = net.node_coords[nid]
            ax.scatter(x, y, s=110, facecolors="none", edgecolors="#00B050",
                       linewidths=2.0, zorder=5)

    # Orange dashed — best conductor upgrade: ทุก segment บน path หม้อแปลง→low-V
    legend_extra: List = []
    if opt.upgrade_options:
        up = opt.upgrade_options[0]
        up_xs, up_ys = [], []
        for _fi, eu, ev in up.upgraded_edges:
            if eu in net.node_coords and ev in net.node_coords:
                xu2, yu2 = net.node_coords[eu]; xv2, yv2 = net.node_coords[ev]
                up_xs += [xu2, xv2, None]; up_ys += [yu2, yv2, None]
        if up_xs:
            ax.plot(up_xs, up_ys, color="#FF6600", lw=4, linestyle="--", zorder=6)
            legend_extra.append(
                Line2D([0],[0], color="#FF6600", lw=3, linestyle="--",
                       label=f"แนะนำ: เพิ่มขนาดสาย →{up.to_size}mm² "
                             f"({len(up.upgraded_edges)} ช่วง)"))

    # Purple dash-dot — best phase addition: draw ALL upgraded edges (ไม่ใช่แค่ edge เดียว)
    if opt.phase_add_options:
        pa = opt.phase_add_options[0]
        pa_xs, pa_ys = [], []
        for _fi, eu, ev in pa.upgraded_edges:
            if eu in net.node_coords and ev in net.node_coords:
                xu2, yu2 = net.node_coords[eu]; xv2, yv2 = net.node_coords[ev]
                pa_xs += [xu2, xv2, None]; pa_ys += [yu2, yv2, None]
        if pa_xs:
            ax.plot(pa_xs, pa_ys, color="#8B00FF", lw=3,
                    linestyle="-.", zorder=6, alpha=0.85)
            legend_extra.append(
                Line2D([0],[0], color="#8B00FF", lw=3, linestyle="-.",
                       label=f"แนะนำ: เพิ่มเฟส→{_PD_LABEL.get(pa.to_pd, 'ABC')} ({len(pa.upgraded_edges)} edges)"))

    # Transformer star
    tx, ty = net.node_coords[net.transformer_node]
    ax.scatter(tx, ty, color=BASE, marker="*", s=480, zorder=7,
               edgecolors="k", linewidths=0.8)
    ax.annotate(f"TR\n{net.facilityid}", (tx, ty),
                xytext=(7, 7), textcoords="offset points",
                fontsize=8, color=BASE, fontweight="bold")

    # Colorbar
    if has_v:
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.58, pad=0.02)
        cbar.set_label("Voltage (V)", fontsize=9)
        y_thr = (opt.min_voltage_v - 150) / 90
        cbar.ax.axhline(y=y_thr, color="red", lw=1.5, linestyle="--")
        cbar.ax.text(1.1, y_thr, f"{opt.min_voltage_v:.0f}V",
                     transform=cbar.ax.transAxes, fontsize=7, color="red", va="center")

    legend_items = [
        Line2D([0],[0], color=BASE, lw=2,        label=net.facilityid),
        Line2D([0],[0], marker="*", color=BASE, markersize=12, lw=0, label="หม้อแปลง"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="red", markersize=8, markeredgewidth=1.5, lw=0,
               label=f"V < {opt.min_voltage_v:.0f}V (หลัง optimize)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#00B050", markersize=8, markeredgewidth=2, lw=0,
               label="มิเตอร์ที่ย้ายเฟส"),
    ] + legend_extra

    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)
    # Title: baseline + all stages
    title = (f"LV Phase/Conductor Analysis  |  {net.facilityid}  "
             f"(rated={net.rated_kva:.0f}kVA)\n"
             f"Baseline: Vmin={b.min_v:.1f}V  TR={b.tr_loading_pct:.1f}%  "
             f"Imbal={b.phase_imbalance_pct:.1f}%  LowV={len(b.low_v_nodes)}nodes")
    if opt.final_result and opt.final_result is not b:
        fr = opt.final_result
        title += (f"\nFinal: Vmin={fr.min_v:.1f}V  TR={fr.tr_loading_pct:.1f}%  "
                  f"Imbal={fr.phase_imbalance_pct:.1f}%  LowV={len(fr.low_v_nodes)}nodes"
                  f"  (สี node = แรงดันหลัง optimize)")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (UTM47N, m)")
    ax.set_ylabel("Northing (UTM47N, m)")
    ax.ticklabel_format(style="plain", axis="both")
    ax.grid(True, alpha=0.22)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Map]   PNG: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Plotly map
# ─────────────────────────────────────────────────────────────────────────────

def draw_interactive_map(opt: LVOptimizer, out_path: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    net   = opt.net
    b     = opt.baseline
    pr    = opt.phase_result
    # แสดงแรงดันจาก final_result (หลัง optimize ทุกขั้นตอน) ถ้ามี
    display_r = opt.final_result if opt.final_result else b
    node_v = {nid: min(ph.values()) for nid, ph in display_r.node_phase_v.items() if ph}
    vmin_c, vmax_c = 150, 240
    cscale = [[0.0, "#d73027"],[0.33, "#fee08b"],[0.66, "#91cf60"],[1.0, "#1a9850"]]

    moved_nids = (
        {m.meter.node_id for m in opt.phase_moves} |
        {m.meter.node_id for m in opt.phase_moves_cu} |
        {m.meter.node_id for m in opt.phase_moves2}
    )
    # Phase colour for edge (by majority conductor PD)
    _phase_color = {7:"#888888", 6:"#0070C0", 5:"#7030A0", 4:"#FF0000",
                    3:"#00B050", 2:"#0070C0", 1:"#7030A0"}

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        specs=[[{"type":"scatter"}],[{"type":"table"}]],
        vertical_spacing=0.04,
    )

    # ── Edges (colored by phase) ─────────────────────────────────────────
    # ใช้ BFS tree edges เท่านั้น — lc_feat_edges มี loop edges ด้วย ทำให้เห็น 2 เส้น
    # สร้าง reverse lookup: (u,v) หรือ (v,u) → feat_idx สำหรับ PHASEDESIGNATION
    _edge_feat: Dict[Tuple[int,int], int] = {}
    for fi, (fu, fv) in net.lc_feat_edges.items():
        _edge_feat[(fu, fv)] = fi
        _edge_feat[(fv, fu)] = fi

    pd_groups: Dict[int, Tuple[List, List]] = defaultdict(lambda: ([],[]))
    for u, v in net.G.edges():          # BFS tree edges only (no duplicates)
        if u not in net.node_coords or v not in net.node_coords:
            continue
        feat_idx = _edge_feat.get((u, v)) or _edge_feat.get((v, u))
        if feat_idx is not None:
            pd = int(get_attr(opt.raw["features"][feat_idx], "PHASEDESIGNATION", 7) or 7)
        else:
            pd = 7
        xs, ys = pd_groups[pd]
        xu, yu = net.node_coords[u]; xv, yv = net.node_coords[v]
        xs += [xu, xv, None]; ys += [yu, yv, None]

    for pd_val, (xs, ys) in pd_groups.items():
        color = _phase_color.get(pd_val, "#888888")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=1.8), opacity=0.6,
            name=f"สาย {_PD_LABEL.get(pd_val,'?')}",
            hoverinfo="skip",
        ), row=1, col=1)

    # ── Nodes coloured by voltage ─────────────────────────────────────────
    nx_v, ny_v, nv_v, nt_v, ns_v = [], [], [], [], []
    nx_n, ny_n, nt_n, ns_n = [], [], [], []

    def _phase_tip(phases):
        return "<br>" + "<br>".join(
            f"V{PHASE_MAP.get(n,'?')} = {v:.1f} V"
            for n, v in sorted(phases.items())) if phases else ""

    for nid, (x, y) in net.node_coords.items():
        kw   = net.node_kw.get(nid, 0.0)
        v    = node_v.get(nid)
        ph   = b.node_phase_v.get(nid, {})
        sz   = 7 + min(kw * 0.6, 14)
        tip  = f"<b>nid={nid}</b>" + _phase_tip(ph) + f"<br>Load={kw:.2f}kW"
        if v is not None:
            nx_v.append(x); ny_v.append(y); nv_v.append(v)
            nt_v.append(tip + f"<br><b>Vmin={v:.1f}V</b>"); ns_v.append(sz)
        else:
            nx_n.append(x); ny_n.append(y)
            nt_n.append(tip); ns_n.append(sz)

    if nx_v:
        fig.add_trace(go.Scatter(
            x=nx_v, y=ny_v, mode="markers",
            marker=dict(color=nv_v, colorscale=cscale, cmin=vmin_c, cmax=vmax_c,
                        size=ns_v, opacity=0.88, line=dict(width=0),
                        colorbar=dict(title="V (V)", thickness=14, len=0.50,
                                      x=1.01, outlinewidth=0.5,
                                      tickvals=list(range(150, 241, 10)))),
            name="โหนด (แรงดัน)", text=nt_v,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)
    if nx_n:
        fig.add_trace(go.Scatter(
            x=nx_n, y=ny_n, mode="markers",
            marker=dict(color="#4472C4", size=ns_n, opacity=0.5),
            name="โหนด (ไม่มีข้อมูล)", text=nt_n,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # ── Low-voltage rings (แสดงเฉพาะโหนดที่ยังต่ำหลัง optimize) ───────────
    lx, ly, lt = [], [], []
    for nid in display_r.low_v_nodes:
        if nid in net.node_coords:
            x, y = net.node_coords[nid]
            v = node_v.get(nid, 0)
            ph = display_r.node_phase_v.get(nid, {})
            lx.append(x); ly.append(y)
            lt.append(f"<b>LowV nid={nid}  Vmin={v:.1f}V</b>" + _phase_tip(ph))
    if lx:
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="markers",
            marker=dict(symbol="circle-open", color="red", size=18, line=dict(width=2.5)),
            name=f"V < {opt.min_voltage_v:.0f}V",
            text=lt, hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # ── Moved meters (green diamond) ─────────────────────────────────────
    mx, my, mt = [], [], []
    for m in opt.phase_moves:
        if m.meter.node_id in net.node_coords:
            x, y = net.node_coords[m.meter.node_id]
            mx.append(x); my.append(y)
            mt.append(f"<b>ย้ายเฟส: {m.meter.peano or 'idx='+str(m.meter.feat_idx)}</b>"
                      f"<br>{PD_TO_PHASE.get(m.from_pd,'?')} → "
                      f"{PD_TO_PHASE.get(m.to_pd,'?')}"
                      f"<br>ΔV={m.delta_min_v:+.1f}V  kW={m.meter.kw:.2f}")
    if mx:
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="markers",
            marker=dict(symbol="circle-open", color="#00B050", size=20,
                        line=dict(width=2.5)),
            name="มิเตอร์ที่ย้ายเฟส",
            text=mt, hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # ── มิเตอร์ที่ถูกกระจายเฟสใหม่หลังเพิ่มเฟสสาย ─────────────────────────
    if opt.applied_phase_add:
        # วงกลมเปิด สีขอบ = เฟสใหม่ที่ต้องย้ายไป (A แดง / B เหลือง / C น้ำเงิน)
        by_pd: Dict[int, Tuple[List, List, List]] = {pd: ([], [], []) for pd in (4, 2, 1)}
        for fi, new_pd in opt.applied_phase_add.meter_moves:
            nid = net.load_feat_nodes.get(fi)
            feat = opt.raw["features"][fi]
            g = feat.get("geometry", {})
            fx, fy = g.get("x"), g.get("y")
            if fx is None or fy is None:
                if nid not in net.node_coords:
                    continue
                fx, fy = net.node_coords[nid]
            old_pd = _meter_current_pd(feat)
            if old_pd == new_pd or new_pd not in by_pd:
                continue                    # เฟสเดิมอยู่แล้ว ไม่ต้องทำอะไรหน้างาน
            xs, ys, ts = by_pd[new_pd]
            xs.append(fx); ys.append(fy)
            ts.append(f"<b>กระจายเฟสใหม่: "
                      f"{str(get_attr(feat, 'PEANO', '') or '').strip() or 'idx='+str(fi)}</b>"
                      f"<br>{PD_TO_PHASE.get(old_pd,'?')} → {PD_TO_PHASE.get(new_pd,'?')}"
                      f"<br>kW={float(get_attr(feat, 'KWP', 0.0) or 0.0):.2f}")
        for pd_new in (4, 2, 1):
            xs, ys, ts = by_pd[pd_new]
            if not xs:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="circle-open", size=16,
                            color=_PHASE_MARK_COLOR[pd_new],
                            line=dict(width=3, color=_PHASE_MARK_COLOR[pd_new])),
                name=f"ย้ายไปเฟส {PD_TO_PHASE[pd_new]} ({len(xs)} ตัว)",
                text=ts, hovertemplate="%{text}<extra></extra>",
            ), row=1, col=1)

    # ── Conductor upgrade: ทุก segment บน path หม้อแปลง→low-V ─────────────
    for up in opt.upgrade_options[:1]:
        cu_xs, cu_ys = [], []
        for _fi, eu, ev in up.upgraded_edges:
            if eu in net.node_coords and ev in net.node_coords:
                xu2, yu2 = net.node_coords[eu]; xv2, yv2 = net.node_coords[ev]
                cu_xs += [xu2, xv2, None]; cu_ys += [yu2, yv2, None]
        if cu_xs:
            tip = (f"<b>แนะนำ: เพิ่มขนาดสาย</b><br>"
                   f"{len(up.upgraded_edges)} ช่วง บน path หม้อแปลง→โหนดแรงดันต่ำ<br>"
                   f"→{up.to_size}mm²<br>"
                   f"ΔV={up.delta_min_v:+.1f}V<extra></extra>")
            fig.add_trace(go.Scatter(
                x=cu_xs, y=cu_ys, mode="lines+markers",
                line=dict(color="#FF6600", width=5, dash="dash"),
                marker=dict(symbol="diamond", size=10, color="#FF6600"),
                name=f"เพิ่มขนาดสาย →{up.to_size}mm² ({len(up.upgraded_edges)} ช่วง)",
                hovertemplate=tip,
            ), row=1, col=1)

    # ── Phase addition: draw ALL upgraded edges (ไม่ใช่แค่ edge เดียว) ─────
    for pa in opt.phase_add_options[:1]:
        pa_xs, pa_ys = [], []
        for _fi, eu, ev in pa.upgraded_edges:
            if eu in net.node_coords and ev in net.node_coords:
                xu2, yu2 = net.node_coords[eu]; xv2, yv2 = net.node_coords[ev]
                pa_xs += [xu2, xv2, None]; pa_ys += [yu2, yv2, None]
        if pa_xs:
            tip = (f"<b>แนะนำ: เพิ่มเฟสสาย→{_PD_LABEL.get(pa.to_pd, 'ABC')}</b><br>"
                   f"{len(pa.upgraded_edges)} edges<br>"
                   f"ย้าย {len(pa.meter_moves)} มิเตอร์<br>"
                   f"ΔV={pa.delta_min_v:+.1f}V<extra></extra>")
            fig.add_trace(go.Scatter(
                x=pa_xs, y=pa_ys, mode="lines",
                line=dict(color="#8B00FF", width=4, dash="dashdot"),
                name=f"เพิ่มเฟส→{_PD_LABEL.get(pa.to_pd, 'ABC')} ({len(pa.upgraded_edges)} edges)",
                hovertemplate=tip,
                opacity=0.85,
            ), row=1, col=1)

    # ── Meter points ──────────────────────────────────────────────────────
    mmx, mmy, mmv, mmt = [], [], [], []
    for feat_idx, node_id in net.load_feat_nodes.items():
        feat = opt.raw["features"][feat_idx]
        g = feat.get("geometry", {})
        fx, fy = g.get("x"), g.get("y")
        if fx is None or fy is None:
            continue
        peano = str(get_attr(feat, "PEANO", "") or "").strip()
        kw    = float(get_attr(feat, "KWP", 0.0) or 0.0)
        v     = node_v.get(node_id)
        tip   = f"<b>Meter {peano or 'nid='+str(node_id)}</b><br>kW={kw:.2f}<br>"
        tip  += f"<b>V={v:.1f}V</b>" if v else "V=—"
        mmx.append(fx); mmy.append(fy)
        mmv.append(v if v else float("nan"))
        mmt.append(tip)

    vx2, vy2, vv2, vt2 = [], [], [], []
    nx2, ny2, nt2       = [], [], []
    for x, y, v, t in zip(mmx, mmy, mmv, mmt):
        if not math.isnan(v):
            vx2.append(x); vy2.append(y); vv2.append(v); vt2.append(t)
        else:
            nx2.append(x); ny2.append(y); nt2.append(t)
    if vx2:
        fig.add_trace(go.Scatter(
            x=vx2, y=vy2, mode="markers",
            marker=dict(symbol="diamond", color=vv2, colorscale=cscale,
                        cmin=vmin_c, cmax=vmax_c, size=9, opacity=0.88,
                        line=dict(width=0.8, color="rgba(0,0,0,0.3)"), showscale=False),
            name="มิเตอร์ลูกค้า", text=vt2,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)
    if nx2:
        fig.add_trace(go.Scatter(
            x=nx2, y=ny2, mode="markers",
            marker=dict(symbol="diamond", color="gray", size=7, opacity=0.45),
            name="มิเตอร์ (ไม่มี V)", text=nt2,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # ── Transformer ───────────────────────────────────────────────────────
    tx, ty = net.node_coords[net.transformer_node]
    fig.add_trace(go.Scatter(
        x=[tx], y=[ty], mode="markers+text",
        marker=dict(symbol="star", size=24, color="#4472C4",
                    line=dict(color="black", width=1.5)),
        text=[f"TR {net.facilityid}"], textposition="top right",
        textfont=dict(size=10, color="#4472C4"),
        name=f"หม้อแปลง {net.facilityid}",
        hovertemplate=(f"<b>TR {net.facilityid}</b><br>"
                       f"rated={net.rated_kva:.0f}kVA<br>"
                       f"({tx:.0f},{ty:.0f})<extra></extra>"),
    ), row=1, col=1)

    # ── Summary table ─────────────────────────────────────────────────────
    rows_hdr = ["ขั้นตอน", "รายละเอียด", "Vmin (V)", "Imbal (%)", "LowV nodes", "ΔV (V)"]
    rows_val: List[List] = []
    rows_clr: List[str]  = []

    rows_val.append(["Baseline", f"TR={b.tr_loading_pct:.1f}%",
                     f"{b.min_v:.1f}", f"{b.phase_imbalance_pct:.1f}",
                     str(len(b.low_v_nodes)), "—"])
    rows_clr.append("#fff2cc")

    if opt.phase_moves and pr:
        rows_val.append([f"Phase Transfer ({len(opt.phase_moves)} moves)",
                         ", ".join(f"{PD_TO_PHASE.get(m.from_pd,'?')}→{PD_TO_PHASE.get(m.to_pd,'?')}"
                                   for m in opt.phase_moves[:4]) + ("…" if len(opt.phase_moves)>4 else ""),
                         f"{pr.min_v:.1f}", f"{pr.phase_imbalance_pct:.1f}",
                         str(len(pr.low_v_nodes)),
                         f"{pr.min_v - b.min_v:+.1f}"])
        rows_clr.append("#e2efda" if pr.min_v > b.min_v else "#fce4d6")

    if opt.applied_upgrade and opt.upgrade_result and opt.upgrade_result.converged:
        _cu = opt.applied_upgrade
        ur = opt.upgrade_result
        rows_val.append([f"CU Apply →{_cu.to_size}mm² ({len(_cu.upgraded_edges)} ช่วง) ✓",
                         f"path ({_cu.edge[0]}→{_cu.edge[1]})  {_cu.n_affected}LowV",
                         f"{ur.min_v:.1f}", f"{ur.phase_imbalance_pct:.1f}",
                         str(len(ur.low_v_nodes)), f"{_cu.delta_min_v:+.1f}"])
        rows_clr.append("#e2efda" if _cu.delta_min_v > 0 else "#fce4d6")

    if opt.phase_moves_cu and opt.phase_result_cu and opt.phase_result_cu.converged:
        pur = opt.phase_result_cu
        ur_ref = opt.upgrade_result
        dv_cu = (pur.min_v - ur_ref.min_v) if ur_ref and ur_ref.converged else 0.0
        rows_val.append([f"PT หลัง CU ({len(opt.phase_moves_cu)} moves)",
                         ", ".join(f"{PD_TO_PHASE.get(m.from_pd,'?')}→{PD_TO_PHASE.get(m.to_pd,'?')}"
                                   for m in opt.phase_moves_cu[:4])
                         + ("…" if len(opt.phase_moves_cu) > 4 else ""),
                         f"{pur.min_v:.1f}", f"{pur.phase_imbalance_pct:.1f}",
                         str(len(pur.low_v_nodes)), f"{dv_cu:+.1f}"])
        rows_clr.append("#e2efda" if dv_cu > 0 else "#fce4d6")

    for up in opt.upgrade_options[:2]:
        r2 = up.result
        marker = " ✓" if (up is opt.applied_upgrade) else ""
        rows_val.append([f"Conductor →{up.to_size}mm² ({len(up.upgraded_edges)} ช่วง){marker}",
                         f"path ({up.edge[0]}→{up.edge[1]})  {up.n_affected}โหนด",
                         f"{r2.min_v:.1f}" if r2.converged else "NG",
                         f"{r2.phase_imbalance_pct:.1f}" if r2.converged else "—",
                         str(len(r2.low_v_nodes)) if r2.converged else "—",
                         f"{up.delta_min_v:+.1f}"])
        rows_clr.append("#e2efda" if up.delta_min_v > 0 else "#fce4d6")

    for pa in opt.phase_add_options[:2]:
        r2 = pa.result
        marker = " ✓" if (pa is opt.applied_phase_add) else ""
        rows_val.append([f"Phase Addition {_PD_LABEL.get(pa.from_pd,'?')}→{_PD_LABEL.get(pa.to_pd, 'ABC')}{marker}",
                         f"{len(pa.upgraded_edges)}edges  {len(pa.meter_moves)}มิเตอร์  "
                         f"Δimbal={pa.delta_imbalance:+.1f}%",
                         f"{r2.min_v:.1f}" if r2.converged else "NG",
                         f"{r2.phase_imbalance_pct:.1f}" if r2.converged else "—",
                         str(len(r2.low_v_nodes)) if r2.converged else "—",
                         f"{pa.delta_min_v:+.1f}"])
        rows_clr.append("#e2efda" if (pa.delta_min_v > 0 or pa.delta_imbalance > 0)
                        else "#fce4d6")

    cols_t = list(zip(*rows_val)) if rows_val else [[] for _ in rows_hdr]
    fig.add_trace(go.Table(
        header=dict(values=rows_hdr,
                    fill_color="#1F497D", font=dict(color="white", size=11),
                    align="center", height=28),
        cells=dict(values=cols_t,
                   fill_color=[rows_clr] * len(rows_hdr),
                   font=dict(size=11), align="center", height=24),
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f"LV Phase/Conductor Analysis  |  {net.facilityid}  "
                  f"(rated={net.rated_kva:.0f}kVA)<br>"
                  f"<sup>Baseline: Vmin={b.min_v:.1f}V  Imbal={b.phase_imbalance_pct:.1f}%  "
                  f"LowV={len(b.low_v_nodes)}"
                  + (f"  →  Final: Vmin={display_r.min_v:.1f}V  "
                     f"Imbal={display_r.phase_imbalance_pct:.1f}%  "
                     f"LowV={len(display_r.low_v_nodes)}"
                     if display_r is not b else "")
                  + f"  |  สีโหนด = แรงดันหลัง optimize  |  Click legend = toggle</sup>"),
            font=dict(size=13),
        ),
        hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        height=1080,
        legend=dict(x=1.07, y=0.99, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#cccccc", borderwidth=1, font=dict(size=11)),
        margin=dict(r=220, t=95, l=70, b=30),
    )
    fig.update_xaxes(title_text="Easting (UTM47N, m)", tickformat="d",
                     showgrid=True, gridcolor="#eeeeee", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Northing (UTM47N, m)", tickformat="d",
                     showgrid=True, gridcolor="#eeeeee", zeroline=False,
                     scaleanchor="x", scaleratio=1, row=1, col=1)

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[Map]   Interactive HTML: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LV Phase & Conductor Optimizer — วิเคราะห์และปรับปรุงระบบ LV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("facilityid", nargs="?", default=None,
                    help="FACILITYID หม้อแปลง เช่น 04-123456 (ถ้าไม่ใส่จะถามอีกครั้ง)")
    ap.add_argument("--min-voltage",   type=float, default=200.0, metavar="V",
                    help="แรงดันต่ำสุดที่ยอมรับ (V)")
    ap.add_argument("--max-imbalance", type=float, default=DEFAULT_MAX_IMBALANCE_PCT, metavar="PCT",
                    help="phase imbalance สูงสุด (%%)")
    ap.add_argument("--json-dir",      default=JSON_DIR,
                    help="folder ที่เก็บไฟล์ JSON")
    ap.add_argument("--out-dir",       default=DEFAULT_OUT,
                    help="folder สำหรับบันทึก Excel/PNG/HTML")
    ap.add_argument("--max-sim-time",  type=float, default=300.0, metavar="SEC",
                    help="เวลาสูงสุด (วินาที) สำหรับ Phase Transfer แต่ละรอบ")
    ap.add_argument("--min-move-gain", type=float, default=MIN_MOVE_GAIN_KW, metavar="KW",
                    help="ย้ายมิเตอร์ 1 ตัวต้องลดความเบี่ยงเบนต่อเฟส (RMS) ได้อย่างน้อยเท่านี้ "
                         "— ตั้งสูงขึ้น = ย้ายน้อยลง")
    ap.add_argument("--max-phase-loading", type=float, default=DEFAULT_MAX_PHASE_LOADING_PCT,
                    metavar="PCT",
                    help="โหลดรายเฟสสูงสุดของหม้อแปลง (%% ของพิกัดเฟส PHASEx_KVA)")
    ap.add_argument("--design-min-length", type=float, default=DESIGN_MIN_COMP_LEN_M,
                    metavar="M",
                    help="Design check: กลุ่มสายที่ยาวตั้งแต่นี้และไม่ครบเฟสหม้อแปลง "
                         "ต้องเพิ่มเฟส แม้ไฟฟ้าจะผ่านเกณฑ์แล้ว")
    ap.add_argument("--design-min-meters", type=int, default=DESIGN_MIN_COMP_METERS,
                    metavar="N",
                    help="Design check: หรือมีมิเตอร์ตั้งแต่จำนวนนี้")
    ap.add_argument("--no-design-check", action="store_true",
                    help="ปิดการตรวจมาตรฐานเฟสสาย (ทำเฉพาะเมื่อมีปัญหาทางไฟฟ้า)")
    ap.add_argument("--region", default=None,
                    help="GIS region สำหรับดึงข้อมูล (ถ้าไม่ระบุใช้ค่า default ของระบบ)")
    args = ap.parse_args()

    if not args.facilityid:
        args.facilityid = input("กรุณากรอก FACILITYID (เช่น 04-123456): ").strip()
    if not args.facilityid:
        print("[!] ไม่ได้ระบุ FACILITYID — ยกเลิก")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    fid  = re.sub(r"[^A-Za-z0-9]", "_", args.facilityid)
    stem = out_dir / f"phase_opt_{fid}_{ts}"

    optimizer = LVOptimizer(
        facilityid=args.facilityid,
        min_voltage_v=args.min_voltage,
        max_imbalance_pct=args.max_imbalance,
        json_dir=args.json_dir,
        max_sim_time_s=args.max_sim_time,
        min_move_gain_kw=args.min_move_gain,
        max_phase_loading_pct=args.max_phase_loading,
        design_check=not args.no_design_check,
        design_min_len_m=args.design_min_length,
        design_min_meters=args.design_min_meters,
        region=args.region,
    )
    optimizer.run()

    if optimizer.net is None or optimizer.baseline is None:
        print("\n[!] วิเคราะห์ไม่สำเร็จ — ตรวจสอบ JSON / GIS connection")
        sys.exit(1)

    print_console_report(optimizer)
    save_excel_report(optimizer, str(stem) + ".xlsx")
    draw_map(optimizer, str(stem) + ".png")
    draw_interactive_map(optimizer, str(stem) + ".html")

    print(f"\nเสร็จสิ้น — ผลลัพธ์บันทึกที่  {out_dir}")


if __name__ == "__main__":
    main()
