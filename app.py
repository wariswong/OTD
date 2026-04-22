from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import os, shutil
import mysql.connector
from processNew_no_gui import run_process_from_project_folder
from InputJsonApi import run_once_with_facilityid, run_pipeline_for_facilityid
from processNew_no_gui_peanumber import main_pipeline
import logging
from collections import defaultdict
import json
import zipfile
import io
import requests
from dotenv import load_dotenv
import urllib.parse
from flask import send_from_directory
import pandas as pd
import csv
import numpy as np

load_dotenv()

app = Flask(__name__,
            static_folder="output",
            static_url_path="/output")
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './output'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'shp'}

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'odt'
}

app.secret_key = os.getenv("SECRET_KEY")

# -------------------------------------------------------------------
# Constants & Helpers
# -------------------------------------------------------------------
REGION_MAPPING = {
    'A': 'N1', 'B': 'N2', 'C': 'N3',
    'D': 'NE1', 'E': 'NE2', 'F': 'NE3',
    'G': 'C1', 'H': 'C2', 'I': 'C3',
    'J': 'S1', 'K': 'S2', 'L': 'S3',
    'Z': 'Z'
}

def is_admin():
    user = session.get("user", {})
    # For now, we'll allow all for testing
    return True

def get_user_region():
    user = session.get("user", {})
    business_area = user.get("hr_business_area", "")
    if business_area:
        char = business_area[0].upper()
        return REGION_MAPPING.get(char, "NE2")
    return "NE2"

# SSO Config
SSO_AUTH_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/auth"
SSO_TOKEN_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/token"
SSO_USERINFO_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/userinfo"
SSO_LOGOUT_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/logout"

CLIENT_ID = os.getenv("SSO_CLIENT_ID")
CLIENT_SECRET = os.getenv("SSO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SSO_REDIRECT_URI")
REDIRECT_URI_CALLBACK = os.getenv("SSO_REDIRECT_URI_CALLBACK")

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

@app.context_processor
def inject_request():
    return dict(request=request)

@app.route("/login")
def login():
    params = {
        "redirect_uri": REDIRECT_URI_CALLBACK,
        "response_type": "code",
        "scope": "openid profile",
        "client_id": CLIENT_ID
    }
    return redirect(f"{SSO_AUTH_URL}?{requests.compat.urlencode(params)}")

@app.route("/login/callback")
def login_callback():
    code = request.args.get("code")
    if not code:
        return "No code provided", 400
    data = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI_CALLBACK,
        "grant_type": "authorization_code",
    }
    r = requests.post(SSO_TOKEN_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    token_json = r.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return "Failed to get access token", 400
    u = requests.get(SSO_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
    userinfo = u.json()
    session["user"] = userinfo
    return redirect(url_for("transformer_stats"))

@app.route("/logout")
def logout():
    session.clear()
    params = {
        "client_id": CLIENT_ID,
        "post_logout_redirect_uri":REDIRECT_URI,
        "lockout": "true"
    }
    return redirect(f"{SSO_LOGOUT_URL}?{requests.compat.urlencode(params)}")

@app.route("/api/userinfo")
def api_userinfo():
    return jsonify(session.get("user", {}))

def get_db():
    return mysql.connector.connect(**db_config)

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@login_required
def index():
    return redirect(url_for('transformer_stats'))

@app.route('/shape_projects')
@login_required
def shape_projects():
    try:
        user = session.get("user", {})
        employee_id = user.get("hr_employee_id")
        region = get_user_region()
        
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        
        # 1. Get projects
        cur.execute("""
            SELECT p.*, 
                (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id=p.id) AS file_count 
            FROM projects p
            WHERE p.owner_id = %s
            ORDER BY p.created_at DESC
        """, (employee_id,))
        projects = cur.fetchall()
        
        # 2. Get stats count for dashboard
        cur.execute("SELECT COUNT(*) as total FROM transformer_stats WHERE region = %s", (region,))
        res_stats = cur.fetchone()
        stats_count = res_stats['total'] if res_stats else 0
        
        cur.close(); conn.close()
        
        for project in projects:
            output_path = os.path.join("output", str(project["id"]))
            project["has_output"] = os.path.exists(output_path) and len(os.listdir(output_path)) > 0
            
        return render_template('index.html', 
                             projects=projects, 
                             user=user, 
                             stats_count=stats_count, 
                             user_region=region)
    except Exception as e:
        logging.error(f"Error in index route: {e}")
        return "เกิดข้อผิดพลาดในระบบ กรุณาติดต่อผู้ดูแล", 500

@app.route('/projectspeanumber')
@login_required
def projectspeanumber():
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT p.* 
        FROM pea_no_projects p
        WHERE p.owner_id = %s
        ORDER BY p.created_at DESC
    """, (employee_id,))
    projects = cur.fetchall()
    cur.close(); conn.close()
    for project in projects:
        output_path = os.path.join("pea_no_projects","output", str(project["id"]))
        project["has_output"] = os.path.exists(output_path) and len(os.listdir(output_path)) > 0
    return render_template('projectspeanumber.html', projects=projects, user=user)

@app.route('/create')
@login_required
def create():
    user = session.get("user", {})
    return render_template('form.html', mode='create', project={}, user=user)

@app.route('/createPeaNumber')
@login_required
def createPeaNumber():
    user = session.get("user", {})
    return render_template('formPeaNumber.html', mode='create', project={}, user=user)

@app.route('/edit/<int:project_id>')
@login_required
def edit(project_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
    project = cur.fetchone()
    cur.execute("SELECT file_type, filename FROM project_files WHERE project_id = %s", (project_id,))
    files = cur.fetchall()
    cur.close(); conn.close()
    existing_files = defaultdict(list)
    for f in files:
        existing_files[f['file_type']].append(f['filename'])
    if not project:
        return "ไม่พบโปรเจค", 404
    user = session.get("user", {})
    return render_template('form.html', mode='edit', project=project, existing_files=existing_files, user=user)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    project_id = request.form.get('project_id')
    name = request.form['project_name']
    detail = request.form.get('project_detail', '')
    if not name.strip():
        return jsonify({'error': 'ชื่อโปรเจคห้ามว่าง'}), 400
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    conn = get_db()
    cur = conn.cursor()
    if project_id:
        cur.execute("UPDATE projects SET project_name=%s, project_detail=%s WHERE id=%s", (name, detail, project_id))
        cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
    else:
        cur.execute("INSERT INTO projects (project_name, project_detail, owner_id) VALUES (%s, %s, %s)", (name, detail, employee_id))
        project_id = str(cur.lastrowid)
    folder_name = secure_filename(project_id)
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
    os.makedirs(folder_path, exist_ok=True)
    uploaded_files = request.files.getlist('folder_files')
    allowed_exts = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']
    valid_prefixes = ['meter', 'lv', 'mv', 'tr', 'eservice']
    for f in uploaded_files:
        filename = os.path.basename(f.filename)
        ext = os.path.splitext(filename)[1].lower()
        prefix = next((p for p in valid_prefixes if filename.lower().startswith(p)), None)
        if ext in allowed_exts and prefix:
            new_filename = f"{prefix}{ext}"
            filepath = os.path.join(folder_path, new_filename)
            f.save(filepath)
            cur.execute("INSERT INTO project_files(project_id, file_type, filename, filepath) VALUES(%s,%s,%s,%s)", (project_id, prefix, filename, filepath))
    conn.commit()
    cur.close(); conn.close()
    return jsonify({'message': 'อัปโหลดสำเร็จ'}), 200

@app.route('/update', methods=['POST'])
def update():
    project_id = request.form.get('project_id')
    project_name = request.form.get('project_name', '').strip()
    project_detail = request.form.get('project_detail', '')
    if not project_id or not project_name:
        return jsonify({'error': 'project_id และ project_name ต้องระบุ'}), 400
    folder = secure_filename(str(project_id))
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    os.makedirs(folder_path, exist_ok=True)
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE projects SET project_name=%s, project_detail=%s WHERE id=%s", (project_name, project_detail, project_id))
        for key in ['file_meter', 'file_lv', 'file_mv', 'file_eservice', 'file_tr']:
            f = request.files.get(key)
            if f and allowed_file(f.filename):
                file_type = key.replace('file_', '')
                filename = f"{file_type}.shp"
                filepath = os.path.join(folder_path, filename)
                f.save(filepath)
                cursor.execute("SELECT id FROM project_files WHERE project_id=%s AND file_type=%s", (project_id, file_type))
                if cursor.fetchone():
                    cursor.execute("UPDATE project_files SET filename=%s, filepath=%s WHERE project_id=%s AND file_type=%s", (filename, filepath, project_id, file_type))
                else:
                    cursor.execute("INSERT INTO project_files (project_id, file_type, filename, filepath) VALUES (%s,%s,%s,%s)", (project_id, file_type, filename, filepath))
        conn.commit()
        return jsonify({'message': 'อัปเดตข้อมูลสำเร็จ'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close(); conn.close()

@app.route('/delete/<int:project_id>', methods=['POST'])
def delete(project_id):
    folder = secure_filename(str(project_id))
    upload_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    output_folder_path = os.path.join(app.config['OUTPUT_FOLDER'], folder)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
    conn.commit()
    cur.close(); conn.close()
    try:
        if os.path.isdir(upload_folder_path): shutil.rmtree(upload_folder_path)
        if os.path.isdir(output_folder_path): shutil.rmtree(output_folder_path)
    except Exception as e:
        logging.error(f"Delete folder error: {e}")
    return jsonify({'message': 'ลบสำเร็จ'}), 200

@app.route('/map/<int:project_id>', methods=['GET'])
@login_required
def map_view(project_id):
    with open(f"output/{project_id}/results.json", "r", encoding="utf-8") as f:
        result_data = json.load(f)
    user = session.get("user", {})
    return render_template("testmap.html", project=project_id, result=result_data, user=user)

@app.route('/run/<int:project_id>', methods=['POST'])
def run_project(project_id):
    try:
        folder_path = app.config['UPLOAD_FOLDER']
        result = run_process_from_project_folder(project_id, folder_path)
        if result["success"]:
            return jsonify({"message": "ประมวลผลเสร็จสิ้น"})
        return jsonify({"error": "ประมวลผลไม่สำเร็จ"}), 500
    except Exception as e:
        logging.exception("Error in running project")
        return jsonify({'error': str(e)}), 500

@app.route('/reprocess/<int:project_id>', methods=['POST'])
def reprocess_with_index(project_id):
    try:
        sp_index = request.json.get("sp_index", 0)
        folder_path = app.config['UPLOAD_FOLDER']
        result = run_process_from_project_folder(project_id, folder_path, sp_index=sp_index)
        if result["success"]:
            return jsonify({"message": "ประมวลผลเสร็จสิ้น"})
        return jsonify({"error": "ประมวลผลไม่สำเร็จ"}), 500
    except Exception as e:
        logging.exception("Error in reprocessing project")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<int:project_id>')
def download_project_files(project_id):
    folder_path = f'output/{project_id}/downloads'
    if not os.path.exists(folder_path):
        return "ไม่พบโฟลเดอร์ดาวน์โหลด", 404
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'project_{project_id}_results.zip')

@app.route('/createPeaNoProjects', methods=['POST'])
@login_required
def create_pea_no_project():
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    facility_id = request.form.get("facility_id", "").strip()
    project_detail = request.form.get("project_detail", "").strip()
    region = request.form.get("region", "").strip()
    if not facility_id:
        return jsonify({"error": "facility_id ต้องระบุ"}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO pea_no_projects (project_name, project_detail, owner_id, region, created_at) VALUES (%s, %s, %s, %s, NOW())", (facility_id, project_detail, employee_id, region))
        conn.commit()
        project_id = str(cur.lastrowid)
        run_pipeline_for_facilityid(project_id=project_id, facility_id=facility_id, region=region)
        return jsonify({"message": "สร้างโปรเจคสำเร็จ", "project_id": project_id}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close(); conn.close()

@app.route('/runPeaNoProjects/<int:project_id>', methods=['POST'])
@login_required
def run_pea_no_project(project_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT project_name FROM pea_no_projects WHERE id = %s", (project_id,))
        project = cur.fetchone()
        if not project: return jsonify({"error": "ไม่พบโปรเจค"}), 404
        main_pipeline(project_id=str(project_id), facility_id=project["project_name"])
        return jsonify({"success": True, "message": "ประมวลผลเสร็จสิ้น"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cur.close(); conn.close()

@app.route('/peaNoMap/<int:project_id>', methods=['GET'])
@login_required
def pea_no_project_map_view(project_id):
    with open(f"pea_no_projects/output/{project_id}/results.json", "r", encoding="utf-8") as f:
        result_data = json.load(f)
    return render_template("peaNoProjectmap.html", project=project_id, result=result_data, user=session.get("user", {}))

@app.route('/downloadPeaNoProject/<int:project_id>')
def download_pea_no_project_files(project_id):
    folder_path = f'pea_no_projects/output/{project_id}/downloads'
    if not os.path.exists(folder_path): return "ไม่พบโฟลเดอร์ดาวน์โหลด", 404
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), folder_path))
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'project_{project_id}_results.zip')

@app.route('/reprocessPeaNoProject/<int:project_id>', methods=['POST'])
def pea_no_project_reprocess_with_index(project_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    try:
        sp_index = request.get_json(silent=True).get("sp_index", 0)
        cur.execute("SELECT project_name FROM pea_no_projects WHERE id = %s", (project_id,))
        project = cur.fetchone()
        if not project: return jsonify({"error": "ไม่พบโปรเจค"}), 404
        main_pipeline(project_id=str(project_id), facility_id=project["project_name"], sp_index=sp_index)
        return jsonify({"success": True, "message": "ประมวลผลเสร็จสิ้น"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cur.close(); conn.close()

@app.route('/pea_no_project_delete/<int:project_id>', methods=['POST'])
def peaNoProjectDelete(project_id):
    folder = secure_filename(str(project_id))
    input_folder_path = os.path.join("pea_no_projects", "input", folder)
    output_folder_path = os.path.join("pea_no_projects", "output", folder)
    conn = get_db(); cur = conn.cursor()
    cur.execute("DELETE FROM pea_no_projects WHERE id=%s", (project_id,))
    conn.commit(); cur.close(); conn.close()
    try:
        if os.path.isdir(input_folder_path): shutil.rmtree(input_folder_path)
        if os.path.isdir(output_folder_path): shutil.rmtree(output_folder_path)
    except Exception as e:
        logging.error(f"Delete folder error: {e}")
    return jsonify({'message': 'ลบสำเร็จ'}), 200

@app.route('/pea_no_projects/output/<int:project_id>/<path:filename>')
def serve_project_output(project_id, filename):
    base_dir = os.path.join(os.getcwd(), "pea_no_projects", "output", str(project_id))
    return send_from_directory(base_dir, filename)

@app.route('/transformer_stats')
@login_required
def transformer_stats():
    try:
        region = get_user_region()
        page = request.args.get('page', 1, type=int)
        search_query = request.args.get('q', '').strip()
        per_page = 15
        conn = get_db(); cur = conn.cursor(dictionary=True)
        
        # Build dynamic query
        where_clause = "WHERE region = %s"
        params = [region]
        if search_query:
            where_clause += " AND (facility_id LIKE %s OR location LIKE %s)"
            params.append(f"%{search_query}%")
            params.append(f"%{search_query}%")

        # Get total count for pagination
        cur.execute(f"SELECT COUNT(*) as total FROM transformer_stats {where_clause}", params)
        res = cur.fetchone()
        total_rows = res['total'] if res else 0
        total_pages = (total_rows + per_page - 1) // per_page
        
        # Get paginated data
        cur.execute(f"SELECT * FROM transformer_stats {where_clause} ORDER BY facility_id ASC LIMIT %s OFFSET %s", 
                   params + [per_page, (page-1)*per_page])
        stats = cur.fetchall()
        cur.close(); conn.close()
        
        return render_template('transformer_stats.html', 
                             stats=stats, 
                             page=page, 
                             total_pages=total_pages, 
                             total_rows=total_rows, 
                             region=region,
                             search_query=search_query,
                             user=session.get("user", {}))
    except Exception as e:
        import traceback
        logging.error(f"Error in transformer_stats: {e}\n{traceback.format_exc()}")
        return f"Internal Server Error: {str(e)}", 500

@app.route('/admin/upload_stats', methods=['GET', 'POST'])
@login_required
def admin_upload_stats():
    if not is_admin(): return "Access Denied", 403
    if request.method == 'POST':
        file = request.files.get('file')
        target_region = request.form.get('region')
        overwrite = request.form.get('overwrite') == 'true'
        if file and target_region:
            try:
                # Read raw data
                raw_data = file.stream.read()
                
                # Try multiple encodings
                decoded_data = None
                # utf-8-sig handles UTF-8 with BOM, windows-874 is broader than tis-620
                for enc in ["utf-8-sig", "windows-874", "tis-620"]:
                    try:
                        decoded_data = raw_data.decode(enc)
                        logging.info(f"Successfully decoded CSV using {enc}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not decoded_data:
                    # Final fallback with replacement
                    decoded_data = raw_data.decode("tis-620", errors='replace')
                    logging.warning("Used fallback decoding (tis-620 with errors='replace')")

                # Use StringIO to feed to pandas
                # engine='python' and sep=None allows auto-detection of delimiter (comma, semicolon, tab, etc.)
                data_io = io.StringIO(decoded_data)
                df = pd.read_csv(data_io, sep=None, engine='python', on_bad_lines='warn')
                
                # Replace Infinity and NaN with None (NULL in MySQL)
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.where(pd.notnull(df), None)
                
                conn = get_db(); cur = conn.cursor()
                if overwrite: 
                    cur.execute("DELETE FROM transformer_stats WHERE region = %s", (target_region,))
                
                insert_sql = """INSERT INTO transformer_stats (region, facility_id, location, feeder_id, rate_kva, kva, nerr, loss, aoj, name, error_msg, opsa_trsummary_len, x, y, rundate, kva_peak, pload_peak, peak_month, pun_peak, vmin_peak, ia_peak, ib_peak, ic_peak, gistag, datetime_peak, subtypecode, lat, lon, ia_rated, ib_rated, ic_rated, pct_ia, pct_ib, pct_ic, pload_kw, i_load, max_len, problem_summary, fix_guideline) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                
                rows = []
                for _, row in df.iterrows():
                    # Use get with default values to handle potential column mismatches
                    rows.append((
                        target_region, 
                        str(row.get('FACILITYID', '')), 
                        str(row.get('LOCATION', '')), 
                        str(row.get('FEEDERID', '')), 
                        row.get('RATEKVA'), 
                        row.get('KVA'), 
                        row.get('NERR'), 
                        row.get('LOSS'), 
                        str(row.get('AOJ', '')), 
                        str(row.get('NAME', '')), 
                        str(row.get('ERROR', '')), 
                        row.get('OPSA.TRSUMMAY.LEN'), 
                        row.get('X'), 
                        row.get('Y'), 
                        str(row.get('RUNDATE', '')), 
                        row.get('KVA_PEAK'), 
                        row.get('PLOAD_PEAK'), 
                        str(row.get('PEAK_MONTH', '')), 
                        row.get('PUN_PEAK'), 
                        row.get('VMIN_PEAK'), 
                        row.get('IA_PEAK'), 
                        row.get('IB_PEAK'), 
                        row.get('IC_PEAK'), 
                        str(row.get('GISTAG', '')), 
                        str(row.get('DATETIME_PEAK', '')), 
                        row.get('SUBTYPECODE'), 
                        row.get('LAT'), 
                        row.get('LON'), 
                        row.get('IA_RATED'), 
                        row.get('IB_RATED'), 
                        row.get('IC_RATED'), 
                        row.get('pct_IA'), 
                        row.get('pct_IB'), 
                        row.get('pct_IC'), 
                        row.get('PLOAD_KW'), 
                        row.get('I_LOAD'), 
                        row.get('MAX_LEN'), 
                        str(row.get('\u0e2a\u0e23\u0e38\u0e1b\u0e1b\u0e31\u0e0d\u0e2b\u0e32', '')), 
                        str(row.get('\u0e41\u0e19\u0e27\u0e17\u0e32\u0e07\u0e01\u0e32\u0e23\u0e41\u0e01\u0e49\u0e44\u0e02', ''))
                    ))
                
                for i in range(0, len(rows), 1000): 
                    cur.executemany(insert_sql, rows[i:i+1000])
                
                conn.commit(); cur.close(); conn.close()
                return jsonify({"message": f"Successfully uploaded {len(rows)} rows for region {target_region}"})
            except Exception as e:
                import traceback
                logging.error(f"Upload error: {e}\n{traceback.format_exc()}")
                return jsonify({"error": f"CSV structure error: {str(e)}"}), 500
    return render_template('upload_stats.html')
    return render_template('upload_stats.html')

@app.route('/admin/migrate_stats')
@login_required
def admin_migrate_stats():
    if not is_admin(): return "Access Denied", 403
    try:
        conn = get_db(); cur = conn.cursor()
        with open('db/create_stats_table.sql', 'r', encoding='utf-8') as f:
            sql = f.read()
        for res in cur.execute(sql, multi=True): pass
        conn.commit(); cur.close(); conn.close()
        return "Migration Successful"
    except Exception as e: return f"Migration Failed: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
