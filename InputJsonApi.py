# FACILITYID → TR XY → BalanceLoad → (join table) → write JSON
# Then: extract lines/meters from JSON → auto snap tolerance → build LV network → validate

import os
import sys
import json
import re
# import threading
import logging
import urllib.request
import urllib.parse
import tkinter as tk
# from tkinter import ttk, messagebox
# from datetime import datetime

# Optional scientific stack (expected in user's environment)
try:
    import numpy as np
except Exception as e:
    print("Missing dependency: numpy")
    raise
try:
    import networkx as nx
except Exception as e:
    print("Missing dependency: networkx")
    raise
try:
    from scipy.spatial import cKDTree
except Exception as e:
    print("Missing dependency: scipy (spatial.cKDTree)")
    raise

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def setup_run_file_logger(facility_id, folder="logs", info_only=False):
    """
    สร้าง FileHandler สำหรับเก็บ log ลงไฟล์ใหม่ทุกครั้งที่เรียกใช้
    - ชื่อไฟล์: {prefix}_YYYYMMDD_HHMMSS.log
    - ถ้า info_only=True จะเก็บเฉพาะ logging.INFO (ไม่เอา WARNING/ERROR)
    - คืนค่า: path ของไฟล์ log

    วิธีใช้: เรียกครั้งเดียวตอนเริ่มโปรแกรม
    """
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{facility_id}_{timestamp}_inputlog.log"
    filepath = os.path.join(folder, filename)

    # สร้าง FileHandler แบบเขียนใหม่ทุกครั้ง
    fh = logging.FileHandler(filepath, mode="w", encoding="utf-8")

    # ถ้าอยากเก็บเฉพาะ INFO จริง ๆ
    if info_only:
        class InfoFilter(logging.Filter):
            def filter(self, record):
                return record.levelno == logging.INFO
        fh.addFilter(InfoFilter())
        fh.setLevel(logging.INFO)
    else:
        # เก็บทุกอย่างตั้งแต่ INFO ขึ้นไป (INFO, WARNING, ERROR, CRITICAL)
        fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    # เอาไปผูกกับ root logger (ตัวเดียวกับที่ basicConfig ใช้)
    root = logging.getLogger()
    root.addHandler(fh)

    logging.info(f"File logging enabled: {filepath}")
    return filepath

# -------------------------------------------------------------------
# Globals (config)
# -------------------------------------------------------------------
TR_LAYER_17    = "http://gisne2.pea.co.th/arcgis/rest/services/PEA/MapServer/17/query"
BALANCE_BASE   = "http://172.16.184.234/arcgis/rest/services/PEA/MapServer/exts/BalanceLoad/BalanceLoad"
PEA_QUERY_BASE = "http://gisne2.pea.co.th/arcgis/rest/services/PEA_QUERY/MapServer"

BAL_TABLE_ID   = 31          # ตารางใน PEA_QUERY
BAL_KEY_TABLE  = "PEAMETER"  # คีย์ฝั่งตาราง
BAL_KEY_BAL    = "PEANO"     # คีย์ฝั่ง BalanceLoad

# -------------------------------------------------------------------
# GeometryServer buffer + Spatial Query (MV layer 26)
# -------------------------------------------------------------------
GEOM_BUFFER_URL = "http://gisne2.pea.co.th/arcgis/rest/services/Utilities/Geometry/GeometryServer/buffer"
MV_LAYER_ID     = 26  # DS_MVconductor ใน PEA_QUERY/MapServer/26

SR_UTM47 = 32647
GEOM_PROJECT_URL = "http://gisne2.pea.co.th/arcgis/rest/services/Utilities/Geometry/GeometryServer/project"

def project_geoms(geoms, in_wkid: int, out_wkid: int):
    # รับ list ของ geometry (ArcGIS JSON: points/paths/rings) แล้วเรียก GeometryServer /project ทีละชนิด
    payload = {"f":"pjson", "inSR": in_wkid, "outSR": out_wkid}
    # รองรับทั้ง point และ polyline
    to_proj = []
    geom_type = None
    if "x" in geoms[0]:        # points
        geom_type = "esriGeometryPoint"
        to_proj = {"geometryType": geom_type, "geometries": geoms}
    else:                      # assume polyline
        geom_type = "esriGeometryPolyline"
        to_proj = {"geometryType": geom_type, "geometries": [{"paths": g["paths"]} for g in geoms]}
    payload["geometries"] = json.dumps(to_proj)
    res = http_json_post(GEOM_PROJECT_URL, payload)
    return res.get("geometries", [])

def reproject_balance_to_32647(balance_json: dict) -> dict:
    # เดา/อ่าน SR ต้นทาง
    sr_in = (balance_json.get("spatialReference") or {}).get("wkid")
    if not sr_in:
        # เดาจากสเกลค่าพิกัด: >= 10,000,000 → 102100
        # (ปลอดภัยสำหรับไทย)
        sample = next((f for f in collect_features(balance_json) if "x" in (f.get("geometry") or {})), None)
        if sample and sample["geometry"]["x"] >= 1e7: sr_in = 102100
        else: sr_in = SR_UTM47
    if int(sr_in) == SR_UTM47:
        return balance_json

    # เดินทุก feature แล้ว project geometry → 32647
    for f in collect_features(balance_json):
        g = f.get("geometry") or {}
        if "x" in g and "y" in g:  # point
            new = project_geoms([{"x": g["x"], "y": g["y"]}], int(sr_in), SR_UTM47)
            if new:
                f["geometry"]["x"], f["geometry"]["y"] = new[0]["x"], new[0]["y"]
        elif "paths" in g:         # polyline
            new = project_geoms([{"paths": g["paths"]}], int(sr_in), SR_UTM47)
            if new:
                f["geometry"]["paths"] = new[0]["paths"]
        elif "rings" in g:         # polygon (เผื่อไว้)
            # ถ้าจำเป็นให้เพิ่มโค้ดกรณี polygon คล้ายกับ polyline
            pass

    # อัปเดต spatialReference ให้ถูกต้อง
    balance_json["spatialReference"] = {"wkid": SR_UTM47}
    return balance_json

def project_point(x: float, y: float, in_wkid: int, out_wkid: int) -> tuple[float, float]:
    params = {
        "f": "pjson",
        "inSR": in_wkid,
        "outSR": out_wkid,
        "geometries": json.dumps({
            "geometryType": "esriGeometryPoint",
            "geometries": [{"x": x, "y": y}]
        })
    }
    data = http_json_post(GEOM_PROJECT_URL, params)
    g = (data or {}).get("geometries") or []
    if not g:
        raise RuntimeError("Project point failed")
    return float(g[0]["x"]), float(g[0]["y"])

def buffer_point_utm47(x: float, y: float, distance_m: float = 200.0) -> dict:
    """
    Buffer จุด (x,y) ใน WKID=32647 → คืน polygon (32647)
    """
    in_geom = {
        "geometryType": "esriGeometryPoint",
        "geometries": [{"x": x, "y": y, "spatialReference": {"wkid": SR_UTM47}}]
    }
    params = {
        "f": "pjson",
        "inSR": SR_UTM47,
        "outSR": SR_UTM47,
        "unionResults": "true",
        "geodesic": "false",
        "distances": distance_m,
        "units": "esriSRUnit_Meter",
        "bufferSR": SR_UTM47,
        "geometries": json.dumps(in_geom)
    }
    data = http_json_post(GEOM_BUFFER_URL, params)
    if "geometries" not in data or not data["geometries"]:
        raise RuntimeError("GeometryServer/buffer: ไม่ได้ geometry กลับมา")
    return data["geometries"][0]

def spatial_query_mv_within(buffer_geom: dict, out_fields="*", where="1=1") -> dict:
    """
    Query PEA_QUERY/26 (DS_MVconductor) โดย geometry=buffer polygon (32647 ทั้งขาเข้า-ขาออก)
    """
    layer_url = f"{PEA_QUERY_BASE}/{MV_LAYER_ID}/query"
    params = {
        "f": "pjson",
        "where": where,
        "geometryType": "esriGeometryPolygon",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": SR_UTM47,
        "outSR": SR_UTM47,
        "returnGeometry": "true",
        "outFields": out_fields,
        "geometry": json.dumps(buffer_geom)
    }
    data = http_json_post(layer_url, params)
    if "error" in data:
        raise RuntimeError(f"MV(26) query error: {data['error'].get('message')}")
    return data

def merge_balance_and_mv_to_file(balance_json_path: str, mv_featureset: dict, out_path: str) -> str:
    base = _load_json(balance_json_path)
    # พยายามดึง SR จาก base ถ้ามี ไม่งั้นกำหนดเป็น 32647
    sr = (base.get("spatialReference") if isinstance(base, dict) else None) or {"wkid": SR_UTM47}

    fc = {"type": "FeatureCollection", "spatialReference": sr, "features": []}

    # 1) ย้ายของเดิมเข้า fc
    if isinstance(base, dict) and "features" in base:
        for f in base["features"]:
            if "type" in f and "geometry" in f:
                fc["features"].append(f)
            else:
                attrs = f.get("attributes") or f.get("properties") or {}
                geom  = f.get("geometry") or {}
                fc["features"].append({"type": "Feature","properties": attrs,"geometry": geom})

    # 2) เติม MV(26)
    for fb in mv_featureset.get("features", []):
        attrs = fb.get("attributes") or {}
        geom  = fb.get("geometry") or {}
        fc["features"].append({"type": "Feature","properties": attrs,"geometry": geom})

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(fc, fp, ensure_ascii=False, indent=2)
    return out_path

# -------------------------------------------------------------------
# HTTP helpers
# -------------------------------------------------------------------
def http_json_get(url: str, timeout=30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def http_json_post(url: str, params: dict, timeout=30) -> dict:
    data = urllib.parse.urlencode(params).encode("utf-8")
    req  = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def build_url(base: str, params: dict) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"

def _sql_eq(field: str, value: str) -> str:
    v = str(value).replace("'", "''")
    return f"{field}='{v}'"

# -------------------------------------------------------------------
# Core fetchers
# -------------------------------------------------------------------
def get_tr_xy_by_facilityid(facilityid: str, timeout=15):
    params = {
        "where": f"FACILITYID='{facilityid}'",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": SR_UTM47,   # ขอ 32647 ตรง ๆ
        "f": "pjson",
    }
    url  = f"{TR_LAYER_17}?{urllib.parse.urlencode(params)}"
    data = http_json_get(url, timeout=timeout)

    feats = (data or {}).get("features") or []
    if not feats:
        raise ValueError(f"ไม่พบ TR สำหรับ FACILITYID={facilityid}")

    g = feats[0].get("geometry") or {}
    if "x" not in g or "y" not in g:
        raise ValueError("แถวที่ได้มาไม่มี geometry (x,y)")

    sr_resp = (data.get("spatialReference") or {}).get("wkid")
    x_raw, y_raw = float(g["x"]), float(g["y"])

    if sr_resp and int(sr_resp) != SR_UTM47:
        # กันเหนียว: project ฝั่งเรา ถ้า service เพิกเฉย outSR
        return project_point(x_raw, y_raw, in_wkid=int(sr_resp), out_wkid=SR_UTM47)

    return x_raw, y_raw


def fetch_balance(x: float, y: float) -> dict:
    # ส่ง geometry เป็น 32647 และย้ำ inSR/outSR ให้คืน 32647 กลับมา
    geom = {"x": x, "y": y, "spatialReference": {"wkid": SR_UTM47}}
    params = {
        "geometry": json.dumps(geom),
        "inSR": SR_UTM47,
        "outSR": SR_UTM47,
        "f": "pjson"
    }
    url = build_url(BALANCE_BASE, params)
    return http_json_get(url)


# -------------------------------------------------------------------
# Balance parse & Table query
# -------------------------------------------------------------------
def collect_features(obj):
    out = []
    def walk(o):
        if isinstance(o, dict):
            if "geometry" in o and ("attributes" in o or "geometryType" in o):
                out.append(o)
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for it in o: walk(it)
    walk(obj); return out

def extract_keys_from_balance(balance_json: dict, key_balance: str):
    s = set()
    for f in collect_features(balance_json):
        attrs = f.get("attributes") or {}
        k = attrs.get(key_balance)
        if k is None:
            continue
        ks = str(k).strip()
        if ks:
            s.add(ks)
    return sorted(s)

def get_table_meta(table_id: int) -> dict:
    url = f"{PEA_QUERY_BASE}/{table_id}?f=pjson"
    meta = http_json_get(url)
    if "error" in meta:
        raise RuntimeError(meta["error"].get("message"))
    return meta

def detect_field_is_string(meta: dict, field_name: str) -> bool:
    for fld in meta.get("fields", []):
        if (fld.get("name","") or "").upper() == field_name.upper():
            return "String" in (fld.get("type") or "")
    return True

_digit_re = re.compile(r"^\s*\d+\s*$")

def split_keys_for_type(keys, target_is_string: bool):
    if target_is_string:
        return [str(k).strip() for k in keys], []
    numeric, skipped = [], []
    for k in keys:
        ks = str(k).strip()
        if _digit_re.match(ks):
            try:
                numeric.append(str(int(ks)))
            except:
                skipped.append(ks)
        else:
            skipped.append(ks)
    return numeric, skipped

def _sql_list(vals, as_string: bool) -> str:
    if as_string:
        return ",".join("'{}'".format(str(v).replace("'", "''")) for v in vals)
    return ",".join(str(v) for v in vals)

def query_table_join_map(keys, table_id: int, key_table: str, init_chunk_size=300) -> dict:
    table_url = f"{PEA_QUERY_BASE}/{table_id}"
    meta = get_table_meta(table_id)
    is_string = detect_field_is_string(meta, key_table)
    usable, skipped = split_keys_for_type(keys, is_string)
    if not usable:
        raise RuntimeError(f"ไม่มีคีย์ที่เข้ากับชนิดของ {key_table} (string={is_string}). skipped={len(skipped)}")

    def _query_chunk(arr):
        where = f"{key_table} IN ({_sql_list(arr, is_string)})"
        params = {"where": where, "outFields": "*", "returnGeometry": "false", "f": "pjson"}
        return http_json_post(table_url.rstrip('/') + "/query", params)

    result = {}
    def process_chunk(arr):
        if not arr: return
        data = _query_chunk(arr)
        if "error" in data:
            if len(arr) > 1:
                mid = len(arr)//2 or 1
                process_chunk(arr[:mid]); process_chunk(arr[mid:])
                return
            raise RuntimeError(f"Table{table_id} query error: {data['error'].get('message')}")
        if data.get("exceededTransferLimit"):
            if len(arr) > 1:
                mid = len(arr)//2 or 1
                process_chunk(arr[:mid]); process_chunk(arr[mid:])
            return
        for fb in data.get("features", []):
            attrs = fb.get("attributes") or {}
            k = attrs.get(key_table)
            if k is not None:
                result[str(k).strip()] = fb

    for i in range(0, len(usable), init_chunk_size):
        process_chunk(usable[i:i+init_chunk_size])
    return result

def join_balance_with_table(balance_json: dict, table_map: dict, key_balance: str, prefix: str = ""):
    joined = []
    for fa in collect_features(balance_json):
        attrsA = dict(fa.get("attributes") or {})
        keyA   = attrsA.get(key_balance)
        keyA   = str(keyA).strip() if keyA is not None else None
        fb     = table_map.get(keyA)
        if fb:
            attrsB = fb.get("attributes") or {}
            for k, v in attrsB.items():
                if prefix:
                    attrsA[f"{prefix}{k}"] = v
                else:
                    if k in attrsA:
                        continue  # do not override existing field
                    attrsA[k] = v
        joined.append({"attributes": attrsA, "geometry": fa.get("geometry")})
    return joined

def run_once_with_facilityid(    
    facilityid: str,
    table_id: int = BAL_TABLE_ID,
    key_table: str = BAL_KEY_TABLE,
    key_balance: str = BAL_KEY_BAL,
    save: bool = True,
    project_id: str = None
) -> dict:
    x, y = get_tr_xy_by_facilityid(facilityid)
    balance = fetch_balance(x, y)
    balance = reproject_balance_to_32647(balance)
    if "error" in balance:
        raise RuntimeError(balance["error"].get("message"))
    keys = extract_keys_from_balance(balance, key_balance)
    if not keys:
        raise RuntimeError(f"ไม่พบ {key_balance} ใน BalanceLoad")
    table_map = query_table_join_map(keys, table_id=table_id, key_table=key_table, init_chunk_size=300)
    joined = join_balance_with_table(balance, table_map, key_balance=key_balance, prefix="")

    out_path = None
    if save:
        with open(f"{key_balance.lower()}_list.txt", "w", encoding="utf-8") as fp:
            for k in keys:
                fp.write(k + "\n")

        out_geo = {
            "type": "FeatureCollection",
            "spatialReference": {"wkid": SR_UTM47},   # << สำคัญ
            "features": []
        }
        for f in joined:
            attrs = f.get("attributes") or {}
            geom  = f.get("geometry") or {}
            out_geo["features"].append({
                "type": "Feature",
                "attributes": attrs,
                "properties": attrs,
                "geometry": geom
            })
        base_dir = os.path.join("pea_no_projects", "input", str(project_id))
        os.makedirs(base_dir, exist_ok=True)

        out_path = os.path.join(
            base_dir,
            f"{project_id}_NetworkLV{facilityid}.json"
        )
        # out_path = f"NetworkLV_{facilityid}.json"
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(out_geo, fp, ensure_ascii=False, indent=2)

    return {"x": x, "y": y, "keys": keys, "matched": len(table_map), "out_path": out_path, "joined": joined}

# -------------------------------------------------------------------
# JSON extractors & snapping
# -------------------------------------------------------------------
def _load_json(obj_or_path):
    if isinstance(obj_or_path, dict):
        return obj_or_path
    with open(obj_or_path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def _collect_features(obj):
    """
    เดิน recursive ทั้ง JSON แล้วรวบรวมทุก dict ที่เป็น Feature
    (type == 'Feature') ไม่ว่าซ่อนอยู่ลึกแค่ไหน
    """
    feats = []

    def walk(x):
        if isinstance(x, dict):
            t = str(x.get("type") or x.get("TYPE") or "").lower()
            if t == "feature":
                feats.append(x)
            # common containers
            for key in ("features", "Features", "FEATURES", "items", "data", "layers", "featureSet"):
                if key in x and isinstance(x[key], list):
                    for it in x[key]:
                        walk(it)
            # continue walk all values
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return feats

def _is_meter(attrs: dict) -> bool:
    tag = str(attrs.get("TAG") or "")
    return tag.startswith("2244MT") or ("ACCOUNTNUMBER" in attrs) or (attrs.get("SUBTYPECODE") == 1)

def _phase_bits_to_str(pd_val) -> str:
    try: v = int(pd_val)
    except Exception: v = 7
    s = []
    if v & 1: s.append('A')
    if v & 2: s.append('B')
    if v & 4: s.append('C')
    return "".join(s) if s else "ABC"

def _find_kw_value(attrs: dict) -> float:
    v = attrs.get("KWP")
    if v is None:
        return 0.0
    try:
        return float(str(v).replace(",", "").strip())
    except:
        return 0.0

def snap_coordinates_to_tolerance(coords_list, tolerance=0.1):
    if not coords_list:
        return {}
    coords_array = np.array(coords_list)
    tree = cKDTree(coords_array)
    pairs = tree.query_pairs(tolerance)
    coord_mapping = {}
    processed = set()
    for i, coord in enumerate(coords_list):
        if i in processed: continue
        nearby_indices = [i]
        for a, b in pairs:
            if a == i and b not in processed:
                nearby_indices.append(b)
            elif b == i and a not in processed:
                nearby_indices.append(a)
        representative = coords_list[i]
        for idx in nearby_indices:
            coord_mapping[coords_list[idx]] = representative
            processed.add(idx)
    for i, coord in enumerate(coords_list):
        if coord not in coord_mapping:
            coord_mapping[coord] = coord
    return coord_mapping

def analyze_coordinate_distribution(coords_list):
    if not coords_list:
        return {}
    coords_array = np.array(list(set(coords_list)))
    if len(coords_array) <= 1:
        return {'count': len(coords_array)}
    tree = cKDTree(coords_array)
    distances, _ = tree.query(coords_array, k=2)
    min_distances = distances[:, 1]
    suggested_tolerance = np.percentile(min_distances, 10) / 2
    return {'count': len(coords_array), 'suggested_tolerance': suggested_tolerance}

def find_optimal_tolerance(coords_list, min_tolerance=1e-7, max_tolerance=1e-3,
                           target_reduction_ratio=0.98, max_iterations=20):
    if not coords_list:
        logging.warning("Empty coordinates list provided to find_optimal_tolerance")
        return min_tolerance
    unique_coords = list(set(coords_list))
    original_count = len(unique_coords)
    if original_count < 100:
        return min_tolerance
    low, high = min_tolerance, max_tolerance
    best_tolerance, best_count = min_tolerance, original_count
    it = 0
    while it < max_iterations and (high - low) > min_tolerance:
        mid = (low + high)/2
        snap_map = snap_coordinates_to_tolerance(unique_coords, mid)
        snapped_count = len(set(snap_map.values()))
        ratio = snapped_count / original_count
        if abs(ratio - target_reduction_ratio) < 0.01:
            best_tolerance = mid; best_count = snapped_count; break
        if ratio > target_reduction_ratio: low = mid
        else: high = mid
        if abs(ratio - target_reduction_ratio) < abs(best_count/original_count - target_reduction_ratio):
            best_tolerance = mid; best_count = snapped_count
        it += 1
    return best_tolerance

def auto_determine_snap_tolerance(meter_locations, lv_lines, mv_lines,
                                  reduction_ratio=0.98, use_analysis=True):
    all_coords = []
    if meter_locations is not None and len(meter_locations) > 0:
        all_coords.extend([(loc[0], loc[1]) for loc in meter_locations])
    for line in lv_lines:
        all_coords.extend([(x, y) for x, y in zip(line['X'], line['Y'])
                           if not (np.isnan(x) or np.isnan(y))])
    for line in mv_lines:
        all_coords.extend([(x, y) for x, y in zip(line['X'], line['Y'])
                           if not (np.isnan(x) or np.isnan(y))])
    if not all_coords:
        logging.warning("No coordinates found for tolerance calculation")
        return 0.000002
    if use_analysis:
        stats = analyze_coordinate_distribution(all_coords)
        if 'suggested_tolerance' in stats:
            min_tol = stats['suggested_tolerance'] / 10
            max_tol = stats['suggested_tolerance'] * 10
        else:
            min_tol, max_tol = 1e-7, 1e-3
    else:
        min_tol, max_tol = 1e-7, 1e-3
    return find_optimal_tolerance(all_coords, min_tol, max_tol, reduction_ratio)

## Extract from json ##

def extractLineData_json(json_input, snap_tolerance=None):
    data  = _load_json(json_input)
    feats = data.get("features", [])

    raw_lines, all_coords = [], []

    for f in feats:
        attrs = f.get("attributes") or {}
        tag = str(attrs.get("TAG", "")).strip().upper()
        if "LC" not in tag:
            continue   # กรองให้เหลือเฉพาะ TAG ที่มี LC

        # รองรับ SUBTYPECODE
        subval = attrs.get("SUBTYPECODE", attrs.get("SUBTYPECOD", None))
        try:
            if int(subval) != 1:
                continue
        except Exception:
            continue

        g = f.get("geometry") or {}
        for key in ("paths", "rings"):
            parts = g.get(key)
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, list) and len(part) >= 2:
                        pts = []
                        for p in part:
                            if isinstance(p, (list, tuple)) and len(p) >= 2:
                                try:
                                    x, y = float(p[0]), float(p[1])
                                    if not (np.isnan(x) or np.isnan(y)):
                                        pts.append((x, y))
                                except Exception:
                                    pass
                        if len(pts) >= 2:
                            raw_lines.append(pts)
                            all_coords.extend(pts)

    # auto snap tol ถ้าไม่ส่งมา
    if snap_tolerance is None:
        tmp_lv = [{"X":[x for x,_ in pts], "Y":[y for _,y in pts]} for pts in raw_lines]
        snap_tolerance = auto_determine_snap_tolerance(np.array([]), tmp_lv, [], 0.98, True)

    logging.info(f"[JSON] snap_tolerance={snap_tolerance:.10f}")

    snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)

    linesX, linesY, lines = [], [], []
    for pts in raw_lines:
        snapped = [snap_map.get(t, t) for t in pts]
        xs = [t[0] for t in snapped]
        ys = [t[1] for t in snapped]
        linesX.append(xs)
        linesY.append(ys)
        lines.append({"X": xs, "Y": ys})

    return linesX, linesY, lines, snap_tolerance, snap_map


def extractMeterData_json(json_input, default_voltage: float = 230.0,
                          drop_zero_load = False,
                          dedup = False):
    data  = _load_json(json_input)
    feats = data.get("features", [])

    meter_xy, voltages, totals, phases_s, peanos = [], [], [], [], []

    for f in feats:
    # ใช้เฉพาะ attributes ถ้ามี; ถ้าไม่มีค่อย fallback ไป properties
        attrs = f.get("attributes")
        if attrs is None:
            attrs = f.get("properties") or {}

        tag = str(attrs.get("TAG", "")).strip().upper()
        if "MT" not in tag:
            continue  # นิยาม: TAG ต้องมี "MT"

        # geometry -> (x, y)
        g = f.get("geometry") or {}
        xy = None
        if "x" in g and "y" in g:
            try: xy = (float(g["x"]), float(g["y"]))
            except: xy = None
        else:
            pts = g.get("points")
            if isinstance(pts, list) and len(pts) >= 1 and len(pts[0]) >= 2:
                try: xy = (float(pts[0][0]), float(pts[0][1]))
                except: xy = None
        if xy is None:
            continue
        meter_xy.append(xy)

        # ---- voltage: VOLTAGE only ----
        v = attrs.get("VOLTAGE", default_voltage)
        try: voltages.append(float(v))
        except: voltages.append(float(default_voltage))

        # ---- total kW ----
        totals.append(_find_kw_value(attrs))

        # ---- phase: use PHASE (fallback to bit mask) ----
        pstr = attrs.get("PHASE")
        if pstr:
            phases_s.append(str(pstr).upper())
        else:
            phases_s.append(_phase_bits_to_str(attrs.get("PHASEDESIGNATION", 7)))

        # ---- PEANO ----
        peanos.append("" if attrs.get("PEANO") is None else str(attrs.get("PEANO")))

    # to arrays & clean
    meter_xy        = np.asarray(meter_xy, dtype=float)
    initialVoltages = np.asarray(voltages, dtype=float)
    totalLoads      = np.asarray(totals,   dtype=float)
    phases          = np.asarray([str(p).upper() if p is not None else ""
                                  for p in phases_s], dtype=object)
    peano           = np.asarray(["" if p is None else str(p) for p in peanos], dtype=object)

    initialVoltages = np.where(np.isfinite(initialVoltages), initialVoltages, float(default_voltage))
    totalLoads = np.where(np.isfinite(totalLoads) & (totalLoads > 0), totalLoads, 0.0)

    mask = np.isfinite(meter_xy[:,0]) & np.isfinite(meter_xy[:,1])
    if drop_zero_load:
        mask &= (totalLoads > 0)

    if dedup and np.any(mask):
        seen = set()
        keep_mask = np.zeros(mask.sum(), dtype=bool)
        cand = np.where(mask)[0]
        j = 0
        for i in cand:
            key = (round(float(meter_xy[i,0]), 6),
                   round(float(meter_xy[i,1]), 6),
                   peano[i])
            if key not in seen:
                seen.add(key)
                keep_mask[j] = True
            j += 1
        final_mask = np.zeros_like(mask, dtype=bool)
        final_mask[cand[keep_mask]] = True
        mask = final_mask

    meterLocations  = meter_xy[mask]
    initialVoltages = initialVoltages[mask]
    totalLoads      = totalLoads[mask]
    phases          = phases[mask]
    peano           = peano[mask]

    # phase loads
    phase_loads = {'A': np.zeros(len(totalLoads), dtype=float),
                   'B': np.zeros(len(totalLoads), dtype=float),
                   'C': np.zeros(len(totalLoads), dtype=float)}
    for i, pstr in enumerate(phases):
        connected = [ph for ph in pstr if ph in ('A','B','C')]
        if not connected:
            connected = ['A','B','C']
        share = totalLoads[i] / len(connected) if connected else 0.0
        for ph in connected:
            phase_loads[ph][i] = share

    lowVoltageIndices = initialVoltages < 100
    if np.any(lowVoltageIndices):
        logging.warning(f"Found {np.sum(lowVoltageIndices)} meters with voltage < 100V.")
    logging.info(f"[JSON] Meters kept: {len(meterLocations)} (non-zero loads: {int((totalLoads>0).sum())})")

    return meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases

def extractServiceLines_json(json_input, snap_tolerance=None):
    """
    ดึงเฉพาะเส้น EServiceline:
      - TAG มีคำว่า "EL"
      - SUBTYPECODE (หรือ SUBTYPECOD) = 5 เท่านั้น
    คืนค่า: svcX, svcY, svcLines, snap_tolerance, snap_map
    """
    data  = _load_json(json_input)
    feats = data.get("features", [])

    raw_lines, all_coords = [], []

    # ตัวแปรนับไว้ debug
    cnt_tag_EL = 0
    cnt_tag_EL_sub5 = 0
    cnt_tag_EL_sub5_geom_ok = 0

    for f in feats:
        attrs = f.get("attributes") or {}
        tag = str(attrs.get("TAG") or "").upper()

        # 1) TAG ต้องมีคำว่า "EL"
        if "EL" not in tag:
            continue
        cnt_tag_EL += 1

        # 2) SUBTYPECODE (หรือ SUBTYPECOD) ต้อง = 5
        subval = attrs.get("SUBTYPECODE", attrs.get("SUBTYPECOD", None))
        try:
            if int(subval) != 5:
                continue
        except Exception:
            # ไม่มี / แปลง int ไม่ได้ → ข้าม
            continue
        cnt_tag_EL_sub5 += 1

        # 3) geometry ต้องเป็นเส้น (paths / rings) และมีอย่างน้อย 2 จุดจริง ๆ
        g = f.get("geometry") or {}
        geom_ok_for_this_feature = False

        for key in ("paths", "rings"):
            parts = g.get(key)
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not (isinstance(part, list) and len(part) >= 2):
                    continue

                pts = []
                for p in part:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        try:
                            x, y = float(p[0]), float(p[1])
                            if not (np.isnan(x) or np.isnan(y)):
                                pts.append((x, y))
                        except Exception:
                            pass

                if len(pts) >= 2:
                    raw_lines.append(pts)
                    all_coords.extend(pts)
                    geom_ok_for_this_feature = True

        if geom_ok_for_this_feature:
            cnt_tag_EL_sub5_geom_ok += 1

    logging.info(
        f"[JSON][EService] feats={len(feats)}, "
        f"TAG has 'EL'={cnt_tag_EL}, "
        f"TAG has 'EL' & SUBTYPE=5={cnt_tag_EL_sub5}, "
        f"those with usable geometry={cnt_tag_EL_sub5_geom_ok}, "
        f"raw_lines(parts)={len(raw_lines)}"
    )

    # auto snap tolerance ถ้าไม่ได้ส่งมา
    if snap_tolerance is None:
        tmp_lines = [{"X":[x for x,_ in pts], "Y":[y for _,y in pts]} for pts in raw_lines]
        snap_tolerance = auto_determine_snap_tolerance(
            meter_locations=np.array([]),
            lv_lines=tmp_lines,
            mv_lines=[],
            reduction_ratio=0.98,
            use_analysis=True
        )
    logging.info(f"[JSON][EService] snap_tolerance={snap_tolerance:.10f}")

    snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)

    svcX, svcY, svcLines = [], [], []
    for pts in raw_lines:
        snapped = [snap_map.get(t, t) for t in pts]
        xs = [t[0] for t in snapped]
        ys = [t[1] for t in snapped]
        svcX.append(xs)
        svcY.append(ys)
        svcLines.append({"X": xs, "Y": ys})

    logging.info(f"[JSON][EService] lines_extracted={len(svcLines)}")
    return svcX, svcY, svcLines, snap_tolerance, snap_map


def extractMVLineData_json(json_input, snap_tolerance=None, tag_contains="MC"):
    """
    ดึงเส้น MV จาก JSON:
      - ใช้เฉพาะ data["features"] ชั้นบนสุด (ไม่วิ่ง _collect_features เพื่อกันซ้ำ)
      - กรอง TAG ที่มีคำว่า tag_contains (ดีฟอลต์ "MC")
      - รองรับ geometry: paths / rings (Esri), LineString / MultiLineString (GeoJSON)
      - คืนค่า: mvX, mvY, mvLines, snap_tolerance, snap_map
    """
    # ---------- load ----------
    data = _load_json(json_input)
    feats = data.get("features", [])

    raw_lines   = []
    all_coords  = []
    tag_hits    = 0   # ฟีเจอร์ที่ TAG มี MC
    mv_feat_cnt = 0   # ฟีเจอร์ที่เป็นเส้น MV จริง ๆ (มี geometry เป็นเส้น)

    def is_line_geom(g: dict) -> bool:
        # Esri Polyline / Polygon
        if isinstance(g.get("paths"), list) and g["paths"]:
            return True
        if isinstance(g.get("rings"), list) and g["rings"]:
            return True
        # GeoJSON
        if "coordinates" in g:
            gt = str(g.get("type", "")).lower()
            if gt in ("linestring", "multilinestring"):
                return True
        return False

    def iter_parts_from_geom(g: dict):
        """
        คืน list ของ parts (แต่ละ part = list จุด [x,y]) จาก geometry
        """
        parts = None
        # Esri
        if isinstance(g.get("paths"), list):
            parts = g["paths"]
        elif isinstance(g.get("rings"), list):
            parts = g["rings"]
        # GeoJSON
        if parts is None and "coordinates" in g:
            coords = g["coordinates"]
            gtype  = str(g.get("type", "")).lower()
            if gtype == "linestring" and isinstance(coords, list):
                parts = [coords]
            elif gtype == "multilinestring" and isinstance(coords, list):
                parts = coords
        if not isinstance(parts, list):
            return []
        return parts

    # ---------- main loop: คัด MV ----------
    for f in feats:
        attrs = f.get("attributes") or f.get("properties") or {}
        tag   = str(attrs.get("TAG", "")).strip().upper()

        if tag_contains.upper() not in tag:
            continue

        tag_hits += 1

        g = f.get("geometry") or {}
        if not is_line_geom(g):
            continue

        mv_feat_cnt += 1
        parts = iter_parts_from_geom(g)
        if not parts:
            continue

        for part in parts:
            if not (isinstance(part, list) and len(part) >= 2):
                continue
            pts = []
            for p in part:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        x, y = float(p[0]), float(p[1])
                        if not (np.isnan(x) or np.isnan(y)):
                            pts.append((x, y))
                    except Exception:
                        pass
            if len(pts) >= 2:
                raw_lines.append(pts)
                all_coords.extend(pts)

    # ---------- log จำนวนเส้น MV ที่เจอจาก JSON ตรง ๆ ----------
    logging.info(
        f"[JSON][MV] features={len(feats)}, tag_hits(MC)={tag_hits}, "
        f"mv_line_features={mv_feat_cnt}, mv_raw_lines={len(raw_lines)}"
    )

    if not raw_lines:
        logging.warning("[JSON][MV] No MV lines extracted (raw_lines=0).")
        # คืนค่าโครงสร้างว่าง ๆ แต่ไม่ให้พัง
        mvX = []; mvY = []; mvLines = []
        if snap_tolerance is None:
            snap_tolerance = 0.000002
        snap_map = {}
        return mvX, mvY, mvLines, snap_tolerance, snap_map

    # ---------- auto snap tolerance ----------
    if snap_tolerance is None:
        tmp = [{"X": [x for x, _ in pts], "Y": [y for _, y in pts]} for pts in raw_lines]
        snap_tolerance = auto_determine_snap_tolerance(
            meter_locations=np.array([]),
            lv_lines=tmp,
            mv_lines=[],
            reduction_ratio=0.98,
            use_analysis=True,
        )

    logging.info(f"[JSON][MV] snap_tolerance={snap_tolerance:.10f}")

    # ---------- snap coordinates ----------
    snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)

    mvX, mvY, mvLines = [], [], []
    for pts in raw_lines:
        snapped = [snap_map.get(t, t) for t in pts]
        xs = [t[0] for t in snapped]
        ys = [t[1] for t in snapped]
        mvX.append(xs)
        mvY.append(ys)
        mvLines.append({"X": xs, "Y": ys})

    logging.info(f"[JSON][MV] lines_extracted={len(mvLines)}")

    return mvX, mvY, mvLines, snap_tolerance, snap_map


def get_transformer_capacity_from_json(json_input, facilityid=None, default_pf=0.875):
    """
    คืนค่า (capacity_kVA, capacity_kW, power_factor)
    - ค้นหา feature ที่มี attributes['RATEKVA'] ในไฟล์ JSON
    - ถ้ามี FACILITYID ให้เลือกตัวที่ FACILITYID ตรงก่อน
    """
    data  = _load_json(json_input)
    feats = _collect_features(data)

    cand = []
    for f in feats:
        attrs = (f.get("attributes") or {})
        if attrs.get("RATEKVA") is None:
            continue
        # เผื่อ RATEKVA มาเป็น string มีคอมมา
        try:
            kva = float(str(attrs["RATEKVA"]).replace(",", "").strip())
        except Exception:
            continue

        fid = str(attrs.get("FACILITYID", "")).strip()
        cand.append((fid, kva))

    if not cand:
        raise KeyError("ไม่พบฟีเจอร์ที่มี RATEKVA ใน JSON")

    # ถ้าระบุ facilityid ให้หาอันที่ตรงก่อน
    cap_kva = None
    if facilityid:
        for fid, kva in cand:
            if fid == str(facilityid):
                cap_kva = kva
                break
    if cap_kva is None:
        cap_kva = cand[0][1]  # ตัวแรกที่เจอ

    pf = float(default_pf)
    cap_kw = cap_kva * pf
    return cap_kva, cap_kw, pf

def build_line_length_map_from_json(json_input,
                                    coord_snap_map,
                                    length_field="SHAPE.LEN",
                                    fallback_fields=("Shape_Leng", "SHAPE_Leng"),
                                    unit_factor=1.0):
    """
    คืน dict: {(ptA, ptB): seg_length_m, (ptB, ptA): seg_length_m, ...}
    - json_input: path หรือ dict ของ FeatureCollection
    - coord_snap_map: mapping จากพิกัดจริง -> พิกัดที่สแน็ปแล้ว
    - length_field: ชื่อฟิลด์ความยาวหลัก (เช่น 'SHAPE.LEN')
    - fallback_fields: ชื่อฟิลด์สำรองถ้าไม่เจอ length_field
    - unit_factor: ตัวคูณหน่วย (ปกติ ArcGIS SHAPE.LEN เป็นเมตร -> ใช้ 1.0)
    """
    data  = _load_json(json_input)
    feats = _collect_features(data)
    line_length_map = {}

    def _first_num(attrs, keys):
        for k in (keys if isinstance(keys, (list, tuple)) else [keys]):
            if k in attrs and attrs[k] is not None:
                try:
                    return float(str(attrs[k]).replace(",", "").strip())
                except Exception:
                    pass
        return None

    for f in feats:
        geom = f.get("geometry") or {}
        attrs = f.get("attributes") or {}

        # รองรับทั้ง paths / rings
        parts = None
        if isinstance(geom.get("paths"), list):
            parts = geom["paths"]
        elif isinstance(geom.get("rings"), list):
            parts = geom["rings"]

        if not parts:
            continue

        # ความยาวรวมทั้งเส้นจาก attributes
        L_attr = _first_num(attrs, [length_field, *fallback_fields])
        if L_attr is None:
            # ไม่มีความยาวในแอตทริบิวต์ ข้ามเส้นนี้
            continue
        L_attr = float(L_attr) * float(unit_factor)

        # ปกติหนึ่ง polyline = 1 part, แต่เผื่อหลาย part
        for part in parts:
            if not (isinstance(part, list) and len(part) >= 2):
                continue

            # snap จุดทั้งหมดของ part
            raw_pts = []
            for p in part:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        x, y = float(p[0]), float(p[1])
                        if not (np.isnan(x) or np.isnan(y)):
                            raw_pts.append((x, y))
                    except Exception:
                        pass
            if len(raw_pts) < 2:
                continue

            snapped = [coord_snap_map.get(pt, pt) for pt in raw_pts]

            # คำนวณความยาวเชิงเส้นรวมจากพิกัด (ไว้ถ่วงสัดส่วน)
            seg_euclid = []
            total_euclid = 0.0
            for i in range(1, len(snapped)):
                a, b = snapped[i-1], snapped[i]
                if a == b:
                    seg_euclid.append(0.0)
                    continue
                d = float(np.hypot(a[0]-b[0], a[1]-b[1]))
                seg_euclid.append(d)
                total_euclid += d

            if total_euclid <= 0:
                # กรณีจุดซ้ำจนรวมเป็นศูนย์ ข้าม
                continue

            # กระจายความยาวจากแอตทริบิวต์ลงแต่ละ segment ตามสัดส่วน
            scale = L_attr / total_euclid
            for i in range(1, len(snapped)):
                a, b = snapped[i-1], snapped[i]
                if a == b:
                    continue
                seg_len = seg_euclid[i-1] * scale  # เมตร (ถ้า L_attr เป็นเมตร)
                line_length_map[(a, b)] = seg_len
                line_length_map[(b, a)] = seg_len

    logging.info(f"[LEN] Built per-segment length map from JSON: {len(line_length_map)//2} segments")
    return line_length_map

def get_transformer_xy_from_json(json_input, facilityid=None):
    """
    คืน (x, y) ของหม้อแปลงจาก JSON:
    - เลือก feature ที่มี RATEKVA ก่อน และ/หรือ FACILITYID ตรง (ถ้าส่งมา)
    - ถ้าไม่เจอ ใช้ point ตัวแรกที่พบใน JSON
    หาไม่ได้ -> raise ValueError
    """
    data  = _load_json(json_input)
    feats = _collect_features(data)

    # หา TR ที่น่าใช่ก่อน (มี RATEKVA หรือ FACILITYID ตรง)
    for f in feats:
        attrs = (f.get("attributes") or {})
        g     = (f.get("geometry") or {})
        if "x" in g and "y" in g:
            has_rate = attrs.get("RATEKVA") is not None
            fid_ok   = (facilityid is None) or (str(attrs.get("FACILITYID","")).strip() == str(facilityid))
            if has_rate or fid_ok:
                return (float(g["x"]), float(g["y"]))

    # ไม่เจอ -> เอา point ตัวแรก
    for f in feats:
        g = (f.get("geometry") or {})
        if "x" in g and "y" in g:
            return (float(g["x"]), float(g["y"]))

    raise ValueError("ไม่พบพิกัดหม้อแปลงใน JSON")

def _phase_bits_to_list(v):
    try:
        vi = int(v)
        out = []
        if vi & 1: out.append('A')
        if vi & 2: out.append('B')
        if vi & 4: out.append('C')
        return out or ['A','B','C']
    except Exception:
        s = str(v).upper()
        cand = [c for c in s if c in ('A','B','C')]
        return cand or ['A','B','C']

def build_phase_indexer_from_json(
    json_input,
    candidate_fields=("PHASEDESIG", "PHASEDESIGNATION", "PHASE", "PHASETYPE"),
    tag_prefix_for_lv=None,   # รับได้ None | str | list/tuple ของ prefix
    *,
    sample_step=10.0,         # ความถี่ sampling บนเส้น (หน่วยเดียวกับพิกัดใน JSON)
    k_candidates=20,
    subtype_allow=(1,),       # ใช้ SUBTYPECODE 1 เป็นหลัก (สาย conductor)
    opvolt_max=1000.0,        # กัน MV: ถ้า OPVOLTINT > opvolt_max จะไม่เอา
):
    data = _load_json(json_input)
    feats = data.get("features", [])

    # --- helper: prefix filter ---
    def _tag_ok(tag: str) -> bool:
        if not tag_prefix_for_lv:
            return True
        if isinstance(tag_prefix_for_lv, (list, tuple, set)):
            return any(str(tag).startswith(p) for p in tag_prefix_for_lv)
        return str(tag).startswith(str(tag_prefix_for_lv))

    # --- helper: phase mapping (ให้ตรงกับไฟล์นี้) ---
    # หลักฐานใน JSON:
    #   PHASEDESIGNATION=4 คู่กับ PHASE="A"  -> 4 = A  :contentReference[oaicite:3]{index=3}
    #   PHASEDESIGNATION=2 คู่กับ PHASE="B"  -> 2 = B  :contentReference[oaicite:4]{index=4}
    # ดังนั้น 1 = C, 2 = B, 4 = A, 7 = ABC
    def _phase_to_allowed(phase_val):
        if phase_val is None or phase_val == "":
            return []
        # 숫자/บิต
        try:
            v = int(phase_val)
            bit_map = {
                1: ["C"],
                2: ["B"],
                4: ["A"],
                3: ["B", "C"],
                5: ["C", "A"],
                6: ["A", "B"],
                7: ["A", "B", "C"],
            }
            return bit_map.get(v, [])
        except Exception:
            pass

        # สตริง เช่น "AB" / "ABC" / "A"
        s = str(phase_val).upper()
        allowed = [c for c in ("A", "B", "C") if c in s]
        return allowed

    # --- collect LV polylines ---
    polylines = []
    allowed_list = []

    for f in feats:
        attrs = f.get("attributes") or {}
        tag = str(attrs.get("TAG") or "")
        if not _tag_ok(tag):
            continue

        # subtype filter (ถ้ามี)
        if subtype_allow is not None and len(subtype_allow) > 0:
            st = attrs.get("SUBTYPECODE", None)
            if st is not None and int(st) not in set(int(x) for x in subtype_allow):
                continue

        # กัน MV ด้วย OPVOLTINT
        opv = attrs.get("OPVOLTINT", None)
        try:
            if opv is not None and float(opv) > float(opvolt_max):
                continue
        except Exception:
            pass

        # หา phase field
        phase_val = None
        for cand in candidate_fields:
            if cand in attrs and attrs[cand] not in (None, ""):
                phase_val = attrs[cand]
                break
        allowed = _phase_to_allowed(phase_val)
        if not allowed:
            continue

        geom = f.get("geometry") or {}
        paths = geom.get("paths")
        if not isinstance(paths, list) or not paths:
            continue

        for part in paths:
            if not isinstance(part, list) or len(part) < 2:
                continue
            pts = []
            for p in part:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        x, y = float(p[0]), float(p[1])
                        if not (np.isnan(x) or np.isnan(y)):
                            pts.append((x, y))
                    except Exception:
                        pass
            if len(pts) >= 2:
                polylines.append(pts)
                allowed_list.append(allowed)

    if not polylines:
        return lambda _xy: ["A", "B", "C"]

    # --- sampling points for KDTree ---
    sample_pts = []
    sample_owner = []

    def _sample_segment(a, b, step):
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        L = (dx * dx + dy * dy) ** 0.5
        if L <= 0:
            return [a]
        if step is None or step <= 0:
            return [a, b]
        n = max(1, int(np.floor(L / step)))
        out = []
        for j in range(n + 1):
            t = j / n
            out.append((ax + t * dx, ay + t * dy))
        return out

    for li, pts in enumerate(polylines):
        for a, b in zip(pts[:-1], pts[1:]):
            for sp in _sample_segment(a, b, sample_step):
                sample_pts.append(sp)
                sample_owner.append(li)

    tree = cKDTree(np.asarray(sample_pts, dtype=float))

    # --- exact point-to-polyline distance ---
    def _pt_seg_dist2(p, a, b):
        px, py = p; ax, ay = a; bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        vv = vx * vx + vy * vy
        if vv <= 0.0:
            dx, dy = px - ax, py - ay
            return dx * dx + dy * dy
        t = (wx * vx + wy * vy) / vv
        if t < 0.0:
            cx, cy = ax, ay
        elif t > 1.0:
            cx, cy = bx, by
        else:
            cx, cy = ax + t * vx, ay + t * vy
        dx, dy = px - cx, py - cy
        return dx * dx + dy * dy

    def _pt_poly_min_dist2(p, poly):
        best = float("inf")
        for a, b in zip(poly[:-1], poly[1:]):
            d2 = _pt_seg_dist2(p, a, b)
            if d2 < best:
                best = d2
        return best

    def allowed_for_xy(xy):
        p = np.asarray(xy, dtype=float)
        if p.shape[0] < 2 or np.any(np.isnan(p[:2])):
            return ["A", "B", "C"]
        p2 = (float(p[0]), float(p[1]))

        k = min(int(k_candidates), len(sample_pts))
        _, idxs = tree.query(np.asarray(p2, dtype=float), k=k)
        idxs = np.atleast_1d(idxs).astype(int).tolist()

        # unique candidate polylines
        cand_lines = []
        seen = set()
        for si in idxs:
            li = int(sample_owner[si])
            if li not in seen:
                seen.add(li)
                cand_lines.append(li)

        best_i = None
        best_d2 = float("inf")
        for li in cand_lines:
            d2 = _pt_poly_min_dist2(p2, polylines[li])
            if d2 < best_d2:
                best_d2 = d2
                best_i = li

        if best_i is None:
            return ["A", "B", "C"]
        return allowed_list[best_i] if allowed_list[best_i] else ["A", "B", "C"]

    return allowed_for_xy

# -------------------------------------------------------------------
# Build network & validate
# -------------------------------------------------------------------
def buildLVNetworkWithLoads(lvLines, mvLines, meterLocations, transformerLocation, phase_loads,
                            conductorResistance, conductorReactance=None, *, svcLines=None,
                            use_shape_length=False, lvData=None, length_field="Shape_Leng",
                            snap_tolerance=0.1):
    if svcLines is None:
        svcLines = []
    logging.info(f"Building LV network with coordinate snapping (tolerance={snap_tolerance}m)...")
    G = nx.Graph()
    node_id = 0
    node_mapping = {}
    coord_mapping = {}
    lv_nodes = set()

    def _build_meter_to_service_map(svcLines, snap_map):
        m2svc = {}
        for line in svcLines:
            pts = [(x, y) for x, y in zip(line['X'], line['Y']) if not np.isnan(x)]
            if len(pts) < 2: continue
            p0 = snap_map.get(tuple(pts[0]), tuple(pts[0]))
            p1 = snap_map.get(tuple(pts[-1]), tuple(pts[-1]))
            m2svc[p0] = p1
            m2svc[p1] = p0
        return m2svc

    all_line_coords = []
    for line in lvLines:
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) if not (np.isnan(x) or np.isnan(y))]
        all_line_coords.extend(coords)
    for line in mvLines:
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) if not (np.isnan(x) or np.isnan(y))]
        all_line_coords.extend(coords)
    coord_snap_map = snap_coordinates_to_tolerance(list(set(all_line_coords)), snap_tolerance)

    line_length_map = {}
    if use_shape_length:
        if lvData is None:
            logging.error("lvData (JSON path/dict) must be provided when use_shape_length=True")
            raise ValueError("lvData must be provided when use_shape_length=True")
        # สร้าง mapping จาก JSON (SHAPE.LEN)
        line_length_map = build_line_length_map_from_json(
            json_input=lvData,
            coord_snap_map=coord_snap_map,
            length_field=length_field if length_field else "SHAPE.LEN",
            fallback_fields=("Shape_Leng", "SHAPE_Leng"),
            unit_factor=1.0  # ปรับถ้าหน่วยไม่ใช่เมตร
        )
    def add_line_to_network(line, is_lv=True):
        nonlocal node_id
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) if not (np.isnan(x) or np.isnan(y))]
        if len(coords) < 2: 
            return
        snapped_coords = [coord_snap_map.get(coord, coord) for coord in coords]
        prev_node = None
        prev_coord = None
        for i, coord in enumerate(snapped_coords):
            if coord not in node_mapping:
                node_mapping[coord] = node_id
                coord_mapping[node_id] = coord
                node_id += 1
            current_node = node_mapping[coord]
            if is_lv:
                lv_nodes.add(current_node)

            if prev_node is not None and prev_node != current_node:
                # ใช้ความยาวจาก JSON ถ้ามี ไม่งั้นใช้ Euclidean
                if use_shape_length and (prev_coord, coord) in line_length_map:
                    seg_len = float(line_length_map[(prev_coord, coord)])  # เมตร
                else:
                    seg_len = float(np.hypot(prev_coord[0] - coord[0], prev_coord[1] - coord[1]))  # เมตร

                R = (seg_len / 1000.0) * conductorResistance
                X = (seg_len / 1000.0) * (conductorReactance if conductorReactance is not None else 0.1 * conductorResistance)

                if not G.has_edge(prev_node, current_node):
                    G.add_edge(prev_node, current_node, weight=seg_len, resistance=R, reactance=X)

            prev_node = current_node
            prev_coord = coord


    for line in lvLines: add_line_to_network(line, is_lv=True)
    for line in mvLines: add_line_to_network(line, is_lv=False)

    logging.info("Connecting meters to network (KDTree)…")
    meter_to_service = _build_meter_to_service_map(svcLines, coord_snap_map)

    if lv_nodes:
        lv_list  = sorted(lv_nodes)
        lv_pts  = np.array([coord_mapping[n] for n in lv_list])
        kdt_lv   = cKDTree(lv_pts)
        
    else:
        kdt_lv = None

    svc_endpts = []
    for line in svcLines:
        pts = [(x,y) for x,y in zip(line['X'], line['Y']) if not np.isnan(x)]
        if len(pts) >= 2:
            p0 = coord_snap_map.get(tuple(pts[0]), tuple(pts[0]))
            p1 = coord_snap_map.get(tuple(pts[-1]),tuple(pts[-1]))
            svc_endpts.extend([p0, p1])
    kdt_svc = cKDTree(svc_endpts) if svc_endpts else None

    meterNodes = []
    for idx, m_xy in enumerate(meterLocations):
        meterNode = node_id
        node_mapping[tuple(m_xy)] = meterNode
        coord_mapping[meterNode]  = tuple(m_xy)
        node_id += 1
        G.add_node(meterNode)
        for ph in 'ABC':
            G.nodes[meterNode][f'load_{ph}'] = phase_loads[ph][idx]
        if kdt_lv is None:
            logging.error("No LV nodes to snap meter.")
            continue
        d_lv, i_lv = kdt_lv.query(m_xy)
        lv_node_default = lv_list[i_lv]
        dist_default    = d_lv
        snap_m = coord_snap_map.get(tuple(m_xy), tuple(m_xy))
        use_service = False
        lv_node = lv_node_default
        dist    = dist_default
        if kdt_svc is not None and svc_endpts:
            d_svc, i_svc = kdt_svc.query(snap_m)
            p_near  = svc_endpts[i_svc]
            p_other = meter_to_service.get(p_near)
            if p_other is not None and d_svc <= d_lv:
                d1 = np.hypot(snap_m[0]-p_near[0], snap_m[1]-p_near[1])
                d_lv2, i_lv2 = kdt_lv.query(p_other)
                lv_node = lv_list[i_lv2]
                dist    = d1 + d_lv2
                use_service = True
        R = dist/1000 * conductorResistance
        X = dist/1000 * (conductorReactance if conductorReactance else 0.1*conductorResistance)
        G.add_edge(meterNode, lv_node, weight=dist, resistance=R, reactance=X, is_service=use_service)
        meterNodes.append(meterNode)

    transformerLocationTuple = tuple(transformerLocation)
    snapped_tx_location = coord_snap_map.get(transformerLocationTuple, transformerLocationTuple)
    if snapped_tx_location in node_mapping:
        transformerNode = node_mapping[snapped_tx_location]
    else:
        transformerNode = node_id
        node_mapping[snapped_tx_location] = transformerNode
        coord_mapping[transformerNode] = snapped_tx_location
        G.add_node(transformerNode)
        for ph in 'ABC':
            G.nodes[transformerNode][f'load_{ph}'] = 0.0
        node_id += 1
        if lv_nodes:
            lv_coords_array = lv_pts
            tx_loc_array = np.array(snapped_tx_location)
            distances = np.sqrt(np.sum((lv_coords_array - tx_loc_array)**2, axis=1))
            min_index = np.argmin(distances)
            closest_node = lv_list[min_index]
            min_dist = distances[min_index]
            resistance = min_dist / 1000 * conductorResistance
            reactance = (min_dist / 1000 * conductorReactance) if conductorReactance is not None else 0.1 * resistance
            G.add_edge(transformerNode, closest_node, weight=min_dist, resistance=resistance, reactance=reactance)
        else:
            logging.warning("No LV lines to connect the transformer.")

    for n in G.nodes:
        for ph in ['A','B','C']:
            if f'load_{ph}' not in G.nodes[n]:
                G.nodes[n][f'load_{ph}'] = 0.0

    if not nx.is_connected(G):
        logging.warning("Network is not fully connected after coordinate snapping!")
        components = list(nx.connected_components(G))
        logging.warning(f"Network has {len(components)} connected components")
    else:
        logging.info("Network is fully connected after coordinate snapping")

    problem_edges = []
    for u, v, data in G.edges(data=True):
        if data['weight'] < snap_tolerance:
            problem_edges.append((u, v, data['weight']))
    if problem_edges:
        logging.warning(f"Found {len(problem_edges)} edges shorter than snap tolerance")

    logging.info(f"LV network built successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logging.info(f"Applied coordinate snapping with tolerance {snap_tolerance}m")
    return (G, transformerNode, meterNodes, node_mapping, coord_mapping)

def identify_failed_snap_lines(lines, snap_tolerance, snap_map=None):
    failed_lines = []
    short_lines = []
    isolated_lines = []
    problematic_connections = []
    if snap_map is None:
        all_coords = []
        for line in lines:
            coords = [(x, y) for x, y in zip(line['X'], line['Y'])
                      if not (np.isnan(x) or np.isnan(y))]
            all_coords.extend(coords)
        snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)

    G = nx.Graph()
    node_id_map = {}
    node_counter = 0
    for idx, line in enumerate(lines):
        coords = [(x, y) for x, y in zip(line['X'], line['Y'])
                  if not (np.isnan(x) or np.isnan(y))]
        if len(coords) < 2:
            failed_lines.append({'index': idx, 'type': 'invalid',
                                 'reason': 'Less than 2 valid coordinates',
                                 'line': line})
            continue
        total_length = 0
        for i in range(len(coords) - 1):
            dist = np.hypot(coords[i][0] - coords[i+1][0], coords[i][1] - coords[i+1][1])
            total_length += dist
        if total_length < snap_tolerance * 2:
            short_lines.append({'index': idx, 'length': total_length,
                                'tolerance': snap_tolerance, 'start': coords[0],
                                'end': coords[-1], 'line': line})
        snapped_coords = [snap_map.get(coord, coord) for coord in coords]
        start_snapped = snapped_coords[0]
        end_snapped = snapped_coords[-1]
        if start_snapped == end_snapped and len(coords) > 2:
            problematic_connections.append({'index': idx, 'reason': 'Start and end snap to same point',
                                            'original_start': coords[0],
                                            'original_end': coords[-1],
                                            'snapped_point': start_snapped, 'line': line})
        for i in range(len(snapped_coords)):
            if snapped_coords[i] not in node_id_map:
                node_id_map[snapped_coords[i]] = node_counter
                node_counter += 1
        for i in range(len(snapped_coords) - 1):
            if snapped_coords[i] != snapped_coords[i+1]:
                n1 = node_id_map[snapped_coords[i]]
                n2 = node_id_map[snapped_coords[i+1]]
                G.add_edge(n1, n2)

    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        if len(components) > 1:
            main_component = max(components, key=len)
            for idx, line in enumerate(lines):
                coords = [(x, y) for x, y in zip(line['X'], line['Y'])
                          if not (np.isnan(x) or np.isnan(y))]
                if len(coords) < 2: continue
                snapped_coords = [snap_map.get(coord, coord) for coord in coords]
                line_nodes = set()
                for coord in snapped_coords:
                    if coord in node_id_map:
                        line_nodes.add(node_id_map[coord])
                hits = [c for c in components if line_nodes.intersection(c)]
                comp_size = len(hits[0]) if hits else 0
                isolated_lines.append({'index': idx, 'component_size': comp_size, 'line': line})

    result = {
        'total_lines': len(lines),
        'failed_lines': failed_lines,
        'short_lines': short_lines,
        'problematic_connections': problematic_connections,
        'isolated_lines': isolated_lines,
        'total_issues': len(failed_lines) + len(short_lines) + len(problematic_connections) + len(isolated_lines)
    }
    logging.info(f"Line snap analysis complete: Total={result['total_lines']} Invalid={len(failed_lines)} "
                 f"Short={len(short_lines)} Problematic={len(problematic_connections)} Isolated={len(isolated_lines)}")
    return result

def validate_network_after_snap(G, coord_mapping, meterNodes, transformerNode):
    if G.number_of_nodes() == 0:
        return {
            'is_connected': False,
            'num_components': 0,
            'components': [],
            'unreachable_meters': list(meterNodes),
            'meters_with_long_path': [],
            'duplicate_edges': [],
            'self_loops': [],
            'isolated_nodes': [],
            'summary': {
                'total_nodes': 0,
                'total_edges': 0,
                'total_meters': len(meterNodes),
                'reachable_meters': 0,
                'network_complete': False
            }
        }

    # คำนวน components ทั้งหมด
    components = list(nx.connected_components(G))
    num_components = len(components)

    # หา component ที่มีหม้อแปลง (ตัวหลักที่เราสนใจ)
    tx_comp = None
    for comp in components:
        if transformerNode in comp:
            tx_comp = comp
            break

    if tx_comp is None:
        # กรณีหม้อแปลงไม่อยู่ในกราฟเลย ถือว่า fail ทั้งหมด
        unreachable_meters = list(meterNodes)
        validation_result = {
            'is_connected': False,
            'num_components': num_components,
            'components': components,
            'unreachable_meters': unreachable_meters,
            'meters_with_long_path': [],
            'duplicate_edges': [],
            'self_loops': list(nx.selfloop_edges(G)),
            'isolated_nodes': list(nx.isolates(G)),
        }
        validation_result['summary'] = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'total_meters': len(meterNodes),
            'reachable_meters': 0,
            'network_complete': False
        }
        logging.warning("[VALIDATE] Transformer node is not in any component!")
        return validation_result

    # เดิม: ยังเก็บ is_connected ไว้เป็นข้อมูล (แต่ไม่ใช้ตัดสิน network_complete แล้ว)
    validation_result = {
        'is_connected': nx.is_connected(G),
        'num_components': num_components,
        'components': components,
        'unreachable_meters': [],
        'meters_with_long_path': [],
        'duplicate_edges': [],
        'self_loops': list(nx.selfloop_edges(G)),
        'isolated_nodes': list(nx.isolates(G)),
    }

    # เช็คระยะทางจาก TR → meter เหมือนเดิม
    for meter in meterNodes:
        # ถ้ามิเตอร์ไม่อยู่ใน component เดียวกับหม้อแปลง → unreachable ทันที
        if meter not in tx_comp:
            validation_result['unreachable_meters'].append(meter)
            continue
        try:
            path_length = nx.shortest_path_length(G, transformerNode, meter, weight='weight')
            if path_length > 1000:  # เกิน 1 กม.
                validation_result['meters_with_long_path'].append({
                    'meter': meter,
                    'distance': path_length
                })
        except nx.NetworkXNoPath:
            validation_result['unreachable_meters'].append(meter)

    # หา duplicate edges เหมือนเดิม
    seen = set()
    for u, v in G.edges():
        e = tuple(sorted((u, v)))
        if e in seen:
            validation_result['duplicate_edges'].append(e)
        seen.add(e)

    total_m = len(meterNodes)
    unreachable = len(validation_result['unreachable_meters'])
    reachable = total_m - unreachable

    validation_result['summary'] = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'total_meters': total_m,
        'reachable_meters': reachable,
        'network_complete': (unreachable == 0)
    }

    logging.info(
        f"[VALIDATE] is_connected={validation_result['is_connected']}, "
        f"num_components={validation_result['num_components']}, "
        f"unreachable_meters={unreachable}, "
        f"network_complete={validation_result['summary']['network_complete']}"
    )
    logging.info("Network validation after snap complete.")
    return validation_result


# -------------------------------------------------------------------
# Worker that runs the full pipeline for a FACILITYID
# -------------------------------------------------------------------
def log_pipeline_result(facility_id: str, result: dict):
    """
    log สรุปผลการ run pipeline สำหรับ FACILITYID หนึ่งตัว
    คาดหวังให้ result มี key:
      - out_json
      - issues_total
      - summary  (dict)
      - lv_count
      - mv_count
      - meter_count
    """
    summary = result.get("summary", {}) or {}

    msg_lines = [
        "=== TRACE RESULT ===",
        f"FACILITYID     : {facility_id}",
        f"out_json       : {result.get('out_json')}",
        f"issues_total   : {result.get('issues_total')}",
        f"LV lines (Overhead)      : {result.get('lv_count', 'N/A')}",
        f"MV lines       : {result.get('mv_count', 'N/A')}",
        f"EService lines : {result.get('eservice_count', 'N/A')}",
        f"Meter count    : {result.get('meter_count', 'N/A')}",
        # summary จาก validate_network_after_snap
        f"Total nodes    : {summary.get('total_nodes', 'N/A')}",
        f"Total edges    : {summary.get('total_edges', 'N/A')}",
        f"Total meters   : {summary.get('total_meters', 'N/A')}",
        f"Reachable mtr. : {summary.get('reachable_meters', 'N/A')}",
        f"Network Connection?    : {summary.get('network_complete', 'N/A')}",
    ]
    msg = "\n".join(msg_lines)

    logging.info(msg)
    return msg  # เผื่อเอาไป print หรือแสดงใน GUI ได้ต่อ


def run_pipeline_for_facilityid(facility_id: str, project_id: str):
    logging.info(f"[Balance] Start with FACILITYID={facility_id}")

    # 1) BalanceLoad + Join → เซฟ TRwitmeter{fac}.json (มีแต่จุด/ฟีเจอร์ตาม Balance)
    result = run_once_with_facilityid(        
        facilityid=facility_id,
        table_id=BAL_TABLE_ID,
        key_table=BAL_KEY_TABLE,
        key_balance=BAL_KEY_BAL,
        save=True,
        project_id=project_id,
    )
    base_json = result["out_path"]
    x, y = result["x"], result["y"]
    logging.info(f"[Balance] Base JSON saved: {base_json}")

    # 2) Buffer 200m รอบ TR แล้ว query MV (layer 26) ภายใน buffer
    try:
        buf_geom = buffer_point_utm47(x, y, distance_m=200.0)
        mv_fs    = spatial_query_mv_within(buf_geom, out_fields="*", where="1=1")
        base_dir = os.path.join("pea_no_projects", "input", str(project_id))
        os.makedirs(base_dir, exist_ok=True)

        merged_json = os.path.join(
            base_dir,
            f"{project_id}_NetworkLV{facility_id}_with_MV.json"
        )
        # merged_json = f"NetworkLV{facility_id}_with_MV.json"
        out_json    = merge_balance_and_mv_to_file(base_json, mv_fs, merged_json)
        logging.info(f"[MV] Merged JSON saved: {out_json} (MV features: {len(mv_fs.get('features', []))})")
    except Exception as e:
        logging.warning(f"[MV] Skip MV merge due to error: {e}")
        out_json = base_json  # ใช้ไฟล์เดิมต่อ

    # 3) Extract เส้น + มิเตอร์ จากไฟล์รวม
    lvX, lvY, lvLines, snap_tol, snap_map = extractLineData_json(out_json, snap_tolerance=None)
    meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases = extractMeterData_json(out_json)
    
    # 3.1) Eservice line Extract
    svcX, svcY, svcLines, svc_tol, svc_snap = extractServiceLines_json(out_json)
   
    # 3.2) MV Line
    try:
        mvX, mvY, mvLines, mv_tol, mv_snap = extractMVLineData_json(
            out_json,
            snap_tolerance=None,      # หรือกำหนดเองก็ได้
            
        )
    except Exception as e:
        logging.exception(f"[MV] extractMVLineData_json failed: {e}")
        mvLines = []
        mvX = []; mvY = []
        mv_tol = None; mv_snap = {}

    
    # ถ้ายังไม่มีเส้นเลย ให้บอกชัด ๆ
    if len(lvLines) == 0:
        logging.warning("[Extract] ไม่พบ polyline ในไฟล์หลัง merge — กราฟจะไม่มี LV edges")

    # 4) คำนวณ SNAP_TOLERANCE อัตโนมัติจากข้อมูลที่มี
    SNAP_TOLERANCE = auto_determine_snap_tolerance(
        meterLocations, lvLines, [], reduction_ratio=0.98, use_analysis=True
    )
    logging.info(f"[Balance] SNAP_TOLERANCE={SNAP_TOLERANCE:.8f} m")

    try:
    # out_json คือไฟล์ JSON ที่คุณสร้างไว้ใน pipeline (เช่น TRwitmeter{fac}_with_mv.json หรือ base_json)
        transformerCapacity_kVA, transformerCapacity, powerFactor = \
            get_transformer_capacity_from_json(out_json, facilityid=facility_id, default_pf=0.875)

        logging.info(f"[TR] RATEKVA={transformerCapacity_kVA} kVA  -> capacity≈{transformerCapacity:.2f} kW (pf={powerFactor})")
    except Exception as e:
        logging.error(f"อ่าน RATEKVA จาก JSON ล้มเหลว: {e}")
        return

    # 5) สร้างกราฟ (ตอนนี้มีโอกาสมีเส้นจาก MV(26) ที่ถูก merge)
    G, transformerNode, meterNodes, node_mapping, coord_mapping = buildLVNetworkWithLoads(
        lvLines=lvLines,          # ตอนนี้คือ "ทุกเส้น" จากไฟล์รวม; ถ้าต้องแยก LV/MV ค่อยกรองทีหลัง
        mvLines=mvLines,
        meterLocations=meterLocations,
        transformerLocation=(x, y),
        phase_loads=phase_loads,
        conductorResistance=0.3, conductorReactance=0.08,
        svcLines=[], use_shape_length=True, lvData=out_json,
        length_field="SHAPE.LEN",
        snap_tolerance=SNAP_TOLERANCE
    )

    # 6) ตรวจ issues/summary
    issues  = identify_failed_snap_lines(lvLines, snap_tolerance=SNAP_TOLERANCE, snap_map=snap_map)
    summary = validate_network_after_snap(G, coord_mapping, meterNodes, transformerNode)

    result = {
        "out_json": out_json,
        "issues_total": issues['total_issues'],
        "summary": summary['summary'],
        "lv_count": len(lvLines),
        "mv_count": len(mvLines),
        "eservice_count": len(svcLines),
        "meter_count": len(meterLocations),
    }

    # ถ้าอยาก log ทันทีใน pipeline (จำเป็นต้องรู้ facility_id ด้วย)
    # log_pipeline_result(facility_id, result)

    return result


# # -------------------------------------------------------------------
# # Tkinter GUI
# # -------------------------------------------------------------------
# class BalanceApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Lowvoltage System Trace (FACILITYID → JSON → Network)")
#         self.geometry("520x140")
#         self.resizable(False, False)

#         frm = ttk.LabelFrame(self, text="Run by FACILITYID Transformer)")
#         frm.pack(fill="both", expand=True, padx=10, pady=10)

#         ttk.Label(frm, text="FACILITYID:").grid(row=0, column=0, padx=(8,4), pady=8, sticky="e")
#         self.entry_fac = ttk.Entry(frm, width=24)
#         self.entry_fac.grid(row=0, column=1, padx=(0,8), pady=8, sticky="w")

#         self.btn_run = ttk.Button(frm, text="Trace Run ", command=self.on_run_click)
#         self.btn_run.grid(row=0, column=2, padx=(0,8), pady=8)

#         self.txt = tk.Text(frm, height=4, width=60)
#         self.txt.grid(row=1, column=0, columnspan=3, padx=8, pady=(0,8))
#         self.txt.insert("end", "ใส่ FACILITYID แล้วกด Trace Run\n")

#     def on_run_click(self):
#         fac = self.entry_fac.get().strip()
#         if not fac:
#             messagebox.showwarning("Trace Run", "กรุณากรอก FACILITYID")
#             return
#         self.btn_run.config(state="disabled")
#         self.txt.insert("end", f"Running… FACILITYID={fac}\n"); self.txt.see("end")
#         t = threading.Thread(target=self._worker_safe, args=(fac,), daemon=True)
#         t.start()

#     def _worker_safe(self, fac):
#         try:
#             res = run_pipeline_for_facilityid(fac)

#             # ใช้ฟังก์ชันสรุปผลที่มี msg_lines ตามที่คุณต้องการ
#             msg = log_pipeline_result(fac, res)

#             # แสดงในกล่องข้อความด้านล่างของ GUI
#             self._append(msg + "\n")

#             # แสดง popup แทนข้อความ "รันเสร็จ"
#             messagebox.showinfo("Trace Result", msg)

#         except Exception as e:
#             logging.exception("Balance pipeline failed")
#             messagebox.showerror("BalanceLoad", f"เกิดข้อผิดพลาด:\n{e}")
#             self._append(f"ERROR: {e}\n")
#         finally:
#             self.btn_run.config(state="normal")

#     def _append(self, s: str):
#         self.txt.insert("end", s); self.txt.see("end")

# # -------------------------------------------------------------------
# # CLI entry (optional) & GUI mainloop
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     if len(sys.argv) >= 2:
#         fac = sys.argv[1]
#         res = run_pipeline_for_facilityid(fac)

#         msg = log_pipeline_result(fac, res)
#         print(msg)
#     else:
#         app = BalanceApp()
#         app.mainloop()


