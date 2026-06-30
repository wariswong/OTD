import os
import re
import math
import subprocess
import threading
import logging
from pathlib import Path
from datetime import datetime
from flask import (
    Blueprint, request, render_template, jsonify,
    session, abort, send_file
)

from ..utils.decorators import login_required

shareload_bp = Blueprint('shareload', __name__)

_ROOT = Path(os.path.dirname(__file__)).parent.parent
SHARELOAD_DIR = _ROOT / 'feature_shareload'
FEASIBLE_DIR = SHARELOAD_DIR / 'FEASIBLE'
FAC_RE = re.compile(r'^\d{2}-\d{6}$')
PAIR_RE = re.compile(r'^(\d{2}-\d{6})_(\d{2}-\d{6})$')

_running_jobs = set()
_running_lock = threading.Lock()


# ------------------------------------------------------------------
# Coordinate conversion: WGS84/UTM Zone 47N (EPSG:32647) → lat/lon
# ------------------------------------------------------------------

def _utm32647_to_latlon(x: float, y: float) -> tuple:
    """Convert EPSG:32647 (UTM Zone 47N) easting/northing to (lat, lon) degrees."""
    try:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
        lon, lat = t.transform(x, y)
        return float(lat), float(lon)
    except Exception:
        pass

    # Manual Bowring/Redfearn UTM → geographic (WGS84)
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    k0 = 0.9996
    E0 = 500000.0
    lon0 = math.radians(99.0)   # Zone 47N central meridian

    xr = x - E0
    yr = y          # northern hemisphere: N0 = 0

    M = yr / k0
    e_ = math.sqrt(1 - e2)
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1 - e_) / (1 + e_)

    phi1 = (mu
            + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96) * math.sin(6*mu)
            + (1097*e1**4/512) * math.sin(8*mu))

    sp = math.sin(phi1)
    N1 = a / math.sqrt(1 - e2 * sp**2)
    T1 = math.tan(phi1)**2
    C1 = ep2 * math.cos(phi1)**2
    R1 = a * (1 - e2) / (1 - e2 * sp**2)**1.5
    D = xr / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*ep2) * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*ep2 - 3*C1**2) * D**6/720
    )
    lon = lon0 + (
        D
        - (1 + 2*T1 + C1) * D**3/6
        + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*ep2 + 24*T1**2) * D**5/120
    ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon)


def _ll(x, y):
    """Return [lon, lat] list for ArcGIS (handles None gracefully)."""
    if x is None or y is None:
        return None
    lat, lon = _utm32647_to_latlon(x, y)
    return [round(lon, 7), round(lat, 7)]


# ------------------------------------------------------------------
# Plotly HTML parser
# ------------------------------------------------------------------

def _parse_num_array(s: str) -> list:
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok == 'null':
            out.append(None)
        else:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def _extract_trace(html: str, name: str) -> dict | None:
    pat = re.compile(
        r'"name":"' + re.escape(name) + r'".*?"x":\[([^\]]*)\].*?"y":\[([^\]]*)\]',
        re.DOTALL
    )
    m = pat.search(html)
    if not m:
        return None
    return {'x': _parse_num_array(m.group(1)), 'y': _parse_num_array(m.group(2))}


def _extract_trace_prefix(html: str, prefix: str) -> dict | None:
    pat = re.compile(
        r'"name":"(' + re.escape(prefix) + r'[^"]*)".*?"x":\[([^\]]*)\].*?"y":\[([^\]]*)\]',
        re.DOTALL
    )
    m = pat.search(html)
    if not m:
        return None
    return {
        'name': m.group(1),
        'x': _parse_num_array(m.group(2)),
        'y': _parse_num_array(m.group(3)),
    }


def _lines_to_paths(xs: list, ys: list) -> list:
    """Split null-separated x/y arrays into list of [[lon,lat],...] paths."""
    paths, current = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            if len(current) >= 2:
                paths.append(current)
            current = []
        else:
            pt = _ll(x, y)
            if pt:
                current.append(pt)
    if len(current) >= 2:
        paths.append(current)
    return paths


def _extract_voltage_nodes(html: str) -> list:
    pat = re.compile(
        r'"name":"Nodes \(voltage\)".*?"x":\[([^\]]*)\].*?"y":\[([^\]]*)\]'
        r'.*?"color":\[([^\]]*)\].*?"text":\[("[^"]*"(?:,"[^"]*")*)\]',
        re.DOTALL
    )
    m = pat.search(html)
    if not m:
        return []
    xs = _parse_num_array(m.group(1))
    ys = _parse_num_array(m.group(2))
    vs = _parse_num_array(m.group(3))
    texts = re.findall(r'"([^"]*)"', m.group(4))
    nodes = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x is None:
            continue
        pt = _ll(x, y)
        if not pt:
            continue
        nodes.append({
            'lon': pt[0], 'lat': pt[1],
            'v': vs[i] if i < len(vs) else None,
            'label': texts[i].replace('<br>', '\n').replace('<b>', '').replace('</b>', '')
                     if i < len(texts) else '',
        })
    return nodes


def _extract_all_switches(html: str) -> list:
    pat = re.compile(
        r'"name":"Switch #\d+".*?"x":\[([^\]]*)\].*?"y":\[([^\]]*)\]',
        re.DOTALL
    )
    switches = []
    for m in pat.finditer(html):
        xs = _parse_num_array(m.group(1))
        ys = _parse_num_array(m.group(2))
        pts = [_ll(x, y) for x, y in zip(xs, ys) if x is not None]
        pts = [p for p in pts if p]
        if len(pts) >= 2:
            switches.append(pts)
    return switches


def parse_plotly_to_map_data(html_path: Path, fac_a: str, fac_b: str) -> dict:
    html = html_path.read_text(encoding='utf-8')

    result = {
        'fac_a': fac_a, 'fac_b': fac_b,
        'tr_a': None, 'tr_b': None,
        'lines_a': [], 'lines_b': [],
        'switch_opt': None, 'tie': None,
        'switches_all': [],
        'nodes_voltage': [],
        'meters_a': [], 'meters_b': [],
    }

    tra = _extract_trace(html, f'TR_A ({fac_a})')
    if tra and tra['x']:
        pt = _ll(tra['x'][0], tra['y'][0])
        if pt:
            result['tr_a'] = {'lon': pt[0], 'lat': pt[1], 'fac': fac_a}

    trb = _extract_trace(html, f'TR_B ({fac_b})')
    if trb and trb['x']:
        pt = _ll(trb['x'][0], trb['y'][0])
        if pt:
            result['tr_b'] = {'lon': pt[0], 'lat': pt[1], 'fac': fac_b}

    la = _extract_trace(html, f'TR_A Lines ({fac_a})')
    if la:
        result['lines_a'] = _lines_to_paths(la['x'], la['y'])

    lb = _extract_trace(html, f'TR_B Lines ({fac_b})')
    if lb:
        result['lines_b'] = _lines_to_paths(lb['x'], lb['y'])

    sw = _extract_trace(html, '[OPT] Switch OPEN')
    if sw and len(sw['x']) >= 2:
        pts = [_ll(x, y) for x, y in zip(sw['x'], sw['y']) if x is not None]
        pts = [p for p in pts if p]
        if len(pts) >= 2:
            result['switch_opt'] = {'pts': pts, 'label': 'Switch OPEN (optimal)'}

    tie = _extract_trace_prefix(html, 'Tie ')
    if tie and len(tie['x']) >= 2:
        pts = [_ll(x, y) for x, y in zip(tie['x'], tie['y']) if x is not None]
        pts = [p for p in pts if p]
        if len(pts) >= 2:
            result['tie'] = {'pts': pts, 'label': tie['name']}

    result['switches_all'] = _extract_all_switches(html)
    result['nodes_voltage'] = _extract_voltage_nodes(html)

    for fac, key in [(fac_a, 'meters_a'), (fac_b, 'meters_b')]:
        net = 'TR_A' if fac == fac_a else 'TR_B'
        mt = _extract_trace(html, f'{net} Meters')
        if mt:
            pts = []
            for x, y in zip(mt['x'], mt['y']):
                if x is not None:
                    p = _ll(x, y)
                    if p:
                        pts.append({'lon': p[0], 'lat': p[1]})
            result[key] = pts

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _list_pairs():
    pairs = []
    if not FEASIBLE_DIR.exists():
        return pairs
    for folder in FEASIBLE_DIR.iterdir():
        if not folder.is_dir():
            continue
        m = PAIR_RE.match(folder.name)
        if not m:
            continue
        fac_a, fac_b = m.group(1), m.group(2)
        html_file = folder / f'transfer_{folder.name}.html'
        pairs.append({
            'key': folder.name,
            'fac_a': fac_a,
            'fac_b': fac_b,
            'has_html': html_file.exists(),
            'has_xlsx': (folder / f'transfer_{folder.name}.xlsx').exists(),
            'mtime': datetime.fromtimestamp(folder.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
        })
    return sorted(pairs, key=lambda x: x['mtime'], reverse=True)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@shareload_bp.route('/shareload')
@login_required
def shareload_list():
    user = session.get('user', {})
    pairs = _list_pairs()
    with _running_lock:
        running = set(_running_jobs)
    return render_template('shareload.html', pairs=pairs, running=running, user=user)


@shareload_bp.route('/shareload/run', methods=['POST'])
@login_required
def shareload_run():
    fac_a = request.form.get('fac_a', '').strip()
    fac_b = request.form.get('fac_b', '').strip()

    if not FAC_RE.match(fac_a) or not FAC_RE.match(fac_b):
        return jsonify({'error': 'รูปแบบ FACILITYID ไม่ถูกต้อง (XX-XXXXXX)'}), 400
    if fac_a == fac_b:
        return jsonify({'error': 'FACILITYID A และ B ต้องไม่เป็นตัวเดียวกัน'}), 400

    pair_key = f'{fac_a}_{fac_b}'
    out_dir = FEASIBLE_DIR / pair_key
    html_path = out_dir / f'transfer_{pair_key}.html'

    if html_path.exists():
        return jsonify({'status': 'exists', 'key': pair_key}), 200

    with _running_lock:
        if pair_key in _running_jobs:
            return jsonify({'status': 'running', 'key': pair_key}), 202
        _running_jobs.add(pair_key)

    def _worker():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            script = SHARELOAD_DIR / 'run_web.py'
            proc = subprocess.run(
                ['python', str(script), fac_a, fac_b, str(out_dir)],
                cwd=str(SHARELOAD_DIR),
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                logging.error(f"[shareload] {pair_key} failed:\n{proc.stderr}")
            else:
                logging.info(f"[shareload] {pair_key} done: {proc.stdout.strip()}")
        except Exception:
            logging.exception(f"[shareload] {pair_key} exception")
        finally:
            with _running_lock:
                _running_jobs.discard(pair_key)

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({'status': 'running', 'key': pair_key}), 202


@shareload_bp.route('/shareload/status/<pair_key>')
@login_required
def shareload_status(pair_key):
    if not PAIR_RE.match(pair_key):
        abort(400)
    html_path = FEASIBLE_DIR / pair_key / f'transfer_{pair_key}.html'
    with _running_lock:
        running = pair_key in _running_jobs
    return jsonify({'ready': html_path.exists(), 'running': running})


@shareload_bp.route('/shareload/result/<pair_key>')
@login_required
def shareload_result(pair_key):
    m = PAIR_RE.match(pair_key)
    if not m:
        abort(404)
    fac_a, fac_b = m.group(1), m.group(2)
    folder = FEASIBLE_DIR / pair_key
    user = session.get('user', {})
    return render_template('shareload_result.html',
        pair_key=pair_key, fac_a=fac_a, fac_b=fac_b,
        has_html=(folder / f'transfer_{pair_key}.html').exists(),
        has_xlsx=(folder / f'transfer_{pair_key}.xlsx').exists(),
        user=user,
    )


@shareload_bp.route('/shareload/data/<pair_key>')
@login_required
def shareload_data(pair_key):
    """Return parsed network data as WGS84 JSON for ArcGIS map rendering."""
    m = PAIR_RE.match(pair_key)
    if not m:
        abort(400)
    fac_a, fac_b = m.group(1), m.group(2)
    html_path = FEASIBLE_DIR / pair_key / f'transfer_{pair_key}.html'
    if not html_path.exists():
        abort(404)
    try:
        data = parse_plotly_to_map_data(html_path, fac_a, fac_b)
        return jsonify(data)
    except Exception as e:
        logging.exception(f"[shareload/data] parse error for {pair_key}")
        return jsonify({'error': str(e)}), 500


@shareload_bp.route('/shareload/view/<pair_key>')
@login_required
def shareload_view_html(pair_key):
    """Serve raw Plotly HTML (fallback/download)."""
    if not PAIR_RE.match(pair_key):
        abort(404)
    html_path = FEASIBLE_DIR / pair_key / f'transfer_{pair_key}.html'
    if not html_path.exists():
        abort(404)
    return send_file(str(html_path), mimetype='text/html')


@shareload_bp.route('/shareload/download/<pair_key>')
@login_required
def shareload_download(pair_key):
    if not PAIR_RE.match(pair_key):
        abort(404)
    xlsx_path = FEASIBLE_DIR / pair_key / f'transfer_{pair_key}.xlsx'
    if not xlsx_path.exists():
        abort(404)
    return send_file(str(xlsx_path), as_attachment=True,
                     download_name=f'transfer_{pair_key}.xlsx')
