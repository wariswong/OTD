from flask import Flask, request, jsonify, render_template, redirect, url_for, render_template, send_file, session, url_for, redirect
from werkzeug.utils import secure_filename
import os, shutil
import mysql.connector
from processNew_no_gui import run_process_from_project_folder
import logging
from collections import defaultdict
import json
import zipfile
import io
import requests
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

app = Flask(__name__,
            static_folder="output",      # บอกให้ static folder ชื่อ output
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

@app.route("/login")
def login():
    params = {        
        "redirect_uri": REDIRECT_URI_CALLBACK,
        "response_type": "code",
        "scope": "openid profile",
        "client_id": CLIENT_ID
    }
    # print(f"{SSO_AUTH_URL}?{requests.compat.urlencode(params)}")
    # return render_template('login.html', linkParams=params)
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
    session["user"] = userinfo  # เก็บข้อมูล user
    print(userinfo)
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    params = {        
        "client_id": CLIENT_ID,
        "post_logout_redirect_uri":REDIRECT_URI,
        "lockout": "true"
    }
    # print((f"{SSO_LOGOUT_URL}?{requests.compat.urlencode(params)}"))
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

    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT p.*, 
               (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id=p.id) AS file_count 
        FROM projects p
        WHERE p.owner_id = %s
        ORDER BY p.created_at DESC
    """, (employee_id,))
    projects = cur.fetchall()
    cur.close(); conn.close()

    for project in projects:
        output_path = os.path.join("output", str(project["id"]))
        project["has_output"] = os.path.exists(output_path) and len(os.listdir(output_path)) > 0

    return render_template('index.html', projects=projects, user=user)


    # return render_template('index.html', projects=projects, user=user)

@app.route('/create')
@login_required
def create():
    user = session.get("user", {})
    return render_template('form.html', mode='create', project={}, user=user)

@app.route('/edit/<int:project_id>')
@login_required
def edit(project_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
    project = cur.fetchone()
    # ดึงชื่อไฟล์เดิม
    cur.execute(
        "SELECT file_type, filename FROM project_files WHERE project_id = %s",
        (project_id,)
    )
    files = cur.fetchall()
    cur.close(); conn.close()

    # สร้าง dict { 'meter':'meter.shp', ... }
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
        cur.execute("UPDATE projects SET project_name=%s, project_detail=%s WHERE id=%s",
                    (name, detail, project_id))
        cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
    else:
        cur.execute("INSERT INTO projects (project_name, project_detail, owner_id) VALUES (%s, %s, %s)",
                    (name, detail, employee_id))
        project_id = str(cur.lastrowid)

    folder_name = secure_filename(project_id)
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        conn.rollback()
        cur.close(); conn.close()
        return jsonify({'error': f'ไม่สามารถสร้างโฟลเดอร์: {e}'}), 500

    uploaded_files = request.files.getlist('folder_files')
    print("RECEIVED FILES:")
    for f in uploaded_files:
        print(f.filename)
    allowed_exts = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']
    valid_prefixes = ['meter', 'lv', 'mv', 'tr', 'eservice']

    for f in uploaded_files:
        filename = os.path.basename(f.filename)  # ✅ ชื่อไฟล์อย่างเดียว
        ext = os.path.splitext(filename)[1].lower()
        # prefix = filename.split('.')[0].lower()
        prefix = next((p for p in valid_prefixes if filename.lower().startswith(p)), None)

        # ถ้าเป็นไฟล์นามสกุลที่อนุญาต และมี prefix ที่ตรง
        if ext in allowed_exts and prefix:
            new_filename = f"{prefix}{ext}"
            filepath = os.path.join(folder_path, new_filename)
            f.save(filepath)

            file_type = prefix  # เช่น 'meter', 'lv'
            cur.execute("""
                INSERT INTO project_files(project_id, file_type, filename, filepath)
                VALUES(%s, %s, %s, %s)
            """, (project_id, file_type, filename, filepath))

    conn.commit()
    cur.close(); conn.close()
    return jsonify({'message': 'อัปโหลดสำเร็จ'}), 200


@app.route('/update', methods=['POST'])
def update():
    project_id    = request.form.get('project_id')
    project_name  = request.form.get('project_name', '').strip()
    project_detail= request.form.get('project_detail', '')

    if not project_id or not project_name:
        return jsonify({'error': 'project_id และ project_name ต้องระบุ'}), 400

    folder = secure_filename(str(project_id))
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    os.makedirs(folder_path, exist_ok=True)

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # อัปเดตชื่อและรายละเอียด
        cursor.execute(
            "UPDATE projects SET project_name=%s, project_detail=%s WHERE id=%s",
            (project_name, project_detail, project_id)
        )

        for key in ['file_meter', 'file_lv', 'file_mv', 'file_eservice', 'file_tr']:
            f = request.files.get(key)
            if f and allowed_file(f.filename):
                file_type = key.replace('file_', '')

                # ตั้งชื่อไฟล์เป็น file_type.shp เช่น lv.shp
                filename = f"{file_type}.shp"
                filepath = os.path.join(folder_path, filename)
                f.save(filepath)  # จะทับไฟล์เดิมหากมี

                # ตรวจว่ามี record เดิมมั้ย
                cursor.execute(
                    "SELECT id FROM project_files WHERE project_id=%s AND file_type=%s",
                    (project_id, file_type)
                )
                exists = cursor.fetchone()

                if exists:
                    cursor.execute(
                        "UPDATE project_files SET filename=%s, filepath=%s WHERE project_id=%s AND file_type=%s",
                        (filename, filepath, project_id, file_type)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO project_files (project_id, file_type, filename, filepath) VALUES (%s,%s,%s,%s)",
                        (project_id, file_type, filename, filepath)
                    )

        conn.commit()
        return jsonify({'message': 'อัปเดตข้อมูลสำเร็จ'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        cursor.close()
        conn.close()



@app.route('/delete/<int:project_id>', methods=['POST'])
def delete(project_id):
    # ใช้ project_id เป็นชื่อโฟลเดอร์โดยตรง
    folder = secure_filename(str(project_id))  # เป็น string เสมอ

    upload_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    output_folder_path = os.path.join(app.config['OUTPUT_FOLDER'], folder)

    # เชื่อมต่อ DB และลบข้อมูล
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
    conn.commit()
    cur.close(); conn.close()

    # ลบโฟลเดอร์จริง
    try:
        if os.path.isdir(upload_folder_path):
            shutil.rmtree(upload_folder_path)
        if os.path.isdir(output_folder_path):
            shutil.rmtree(output_folder_path)
    except Exception as e:
        # ถ้ามี error ระหว่างลบ ให้ log ไว้ (ไม่หยุดการทำงานหลัก)
        logging.error(f"เกิดข้อผิดพลาดในการลบโฟลเดอร์: {e}")

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
        print(result)
        if result["success"]:
            return jsonify({"message": "ประมวลผลเสร็จสิ้น"})
        else:
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
        else:
            return jsonify({"error": "ประมวลผลไม่สำเร็จ"}), 500
    except Exception as e:
        logging.exception("Error in reprocessing project")
        return jsonify({'error': str(e)}), 500
    

@app.route('/download/<int:project_id>')
def download_project_files(project_id):
    folder_path = f'output/{project_id}/downloads'

    if not os.path.exists(folder_path):
        return "ไม่พบโฟลเดอร์ดาวน์โหลด", 404

    # สร้าง ZIP ไฟล์ในหน่วยความจำ
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=relative_path)
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'project_{project_id}_results.zip'
    )

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
