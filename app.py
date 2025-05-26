from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os, shutil
import mysql.connector
from process import run_process_from_project_folder
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'shp'}

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'odt'
}

def get_db():
    return mysql.connector.connect(**db_config)

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT p.*, 
               (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id=p.id) AS file_count 
        FROM projects p
        ORDER BY p.created_at DESC
    """)
    projects = cur.fetchall()
    cur.close(); conn.close()
    return render_template('index.html', projects=projects)

@app.route('/create')
def create():
    return render_template('form.html', mode='create', project={})

@app.route('/edit/<int:project_id>')
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
    existing_files = { f['file_type']: f['filename'] for f in files }
    if not project:
        return "ไม่พบโปรเจค", 404
    return render_template('form.html', mode='edit', project=project, existing_files=existing_files)

@app.route('/upload', methods=['POST'])
def upload():
    project_id = request.form.get('project_id')
    name = request.form['project_name']
    detail = request.form.get('project_detail', '')

    if not name.strip():
        return jsonify({'error': 'ชื่อโปรเจคห้ามว่าง'}), 400

    conn = get_db()
    cur = conn.cursor()

    # ถ้าแก้ไข ให้ update
    if project_id:
        cur.execute("UPDATE projects SET project_name=%s, project_detail=%s WHERE id=%s",
                    (name, detail, project_id))
        # ลบไฟล์เก่า เพื่อ replace ใหม่
        cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
    else:
        cur.execute("INSERT INTO projects (project_name, project_detail) VALUES (%s, %s)",
                    (name, detail))
        project_id = str(cur.lastrowid)  # แปลงให้เป็น string ทันที

    # ✅ ย้ายมาสร้างโฟลเดอร์หลังจากได้ project_id แน่นอน
    folder = secure_filename(project_id)
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        conn.rollback()
        cur.close(); conn.close()
        return jsonify({'error': f'ไม่สามารถสร้างโฟลเดอร์ได้: {e}'}), 500

    # บันทึกไฟล์ .shp
    for key in ['file_meter', 'file_lv', 'file_mv', 'file_eservice', 'file_tr']:
        f = request.files.get(key)
        if f and allowed_file(f.filename):
            file_type = key.replace('file_', '')
            filename = f"{file_type}.shp"
            filepath = os.path.join(folder_path, filename)
            f.save(filepath)
            cur.execute("""
                INSERT INTO project_files(project_id, file_type, filename, filepath)
                VALUES(%s, %s, %s, %s)
            """, (project_id, file_type, filename, filepath))

    conn.commit()
    cur.close(); conn.close()
    return jsonify({'message': 'บันทึกสำเร็จ'}), 200

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
    # ดึงชื่อ project_name เพื่อกำหนดโฟลเดอร์
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT project_name FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if row:
        folder = secure_filename(row[0])
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)

        # ลบ DB
        cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
        conn.commit()

        # ลบโฟลเดอร์จริง
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

    cur.close(); conn.close()
    return jsonify({'message':'ลบสำเร็จ'}), 200

@app.route('/run/<int:project_id>', methods=['POST'])
def run_project(project_id):
    try:
        folder_path = app.config['UPLOAD_FOLDER']
        result = run_process_from_project_folder(project_id, folder_path)

        # return jsonify({'message': 'ประมวลผลเสร็จสิ้น', 'result': result})
        return jsonify(result)
    except Exception as e:
        logging.exception("Error in running project")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
