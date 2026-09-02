import os
import json
import logging
import io
import zipfile
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session, send_file, send_from_directory
from werkzeug.utils import secure_filename
from collections import defaultdict

from ..config import Config
from ..database import get_db_connection
from ..utils.decorators import login_required
from ..utils.helpers import get_user_region, allowed_file
from ..services.project_service import ProjectService

# Existing logic imports
from processNew_no_gui import run_process_from_project_folder
from InputJsonApi import run_pipeline_for_facilityid
from optimized_transformer_group_310869 import main_pipeline

projects_bp = Blueprint('projects', __name__)

@projects_bp.route('/')
@login_required
def index():
    return redirect(url_for('stats.transformer_stats'))

@projects_bp.route('/shape_projects')
@login_required
def shape_projects():
    try:
        user = session.get("user", {})
        employee_id = user.get("hr_employee_id")
        region = get_user_region()
        
        projects = ProjectService.get_shape_projects(employee_id)
        
        # Get stats count for dashboard (could be moved to StatsService)
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as total FROM transformer_stats WHERE region = %s", (region,))
        res_stats = cur.fetchone()
        stats_count = res_stats['total'] if res_stats else 0
        cur.close(); conn.close()
        
        return render_template('index.html', 
                             projects=projects, 
                             user=user, 
                             stats_count=stats_count, 
                             user_region=region)
    except Exception as e:
        logging.error(f"Error in shape_projects route: {e}")
        return "เกิดข้อผิดพลาดในระบบ กรุณาติดต่อผู้ดูแล", 500

@projects_bp.route('/projectspeanumber')
@login_required
def projectspeanumber():
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    projects = ProjectService.get_pea_no_projects(employee_id)
    return render_template('projectspeanumber.html', projects=projects, user=user)

@projects_bp.route('/create')
@login_required
def create():
    user = session.get("user", {})
    return render_template('form.html', mode='create', project={}, user=user)

@projects_bp.route('/createPeaNumber')
@login_required
def createPeaNumber():
    user = session.get("user", {})
    return render_template('formPeaNumber.html', mode='create', project={}, user=user)

@projects_bp.route('/edit/<int:project_id>')
@login_required
def edit(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
    project = cur.fetchone()
    
    if not project:
        cur.close(); conn.close()
        return "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง", 403

    cur.execute("SELECT file_type, filename FROM project_files WHERE project_id = %s", (project_id,))
    files = cur.fetchall()
    cur.close(); conn.close()
    
    existing_files = defaultdict(list)
    for f in files:
        existing_files[f['file_type']].append(f['filename'])
        
    return render_template('form.html', mode='edit', project=project, existing_files=existing_files, user=user)

@projects_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    project_id = request.form.get('project_id')
    name = request.form['project_name']
    detail = request.form.get('project_detail', '')
    if not name.strip():
        return jsonify({'error': 'ชื่อโปรเจคห้ามว่าง'}), 400
    
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    conn = get_db_connection()
    cur = conn.cursor()
    
    if project_id:
        # Check ownership
        cur.execute("SELECT id FROM projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
        if not cur.fetchone():
            cur.close(); conn.close()
            return jsonify({'error': 'คุณไม่มีสิทธิ์แก้ไขโปรเจคนี้'}), 403
        cur.execute("UPDATE projects SET project_name=%s, project_detail=%s WHERE id=%s", (name, detail, project_id))
        cur.execute("DELETE FROM project_files WHERE project_id=%s", (project_id,))
    else:
        cur.execute("INSERT INTO projects (project_name, project_detail, owner_id) VALUES (%s, %s, %s)", (name, detail, employee_id))
        project_id = str(cur.lastrowid)
        
    folder_name = secure_filename(str(project_id))
    folder_path = os.path.join(Config.UPLOAD_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    uploaded_files = request.files.getlist('folder_files')
    for f in uploaded_files:
        filename = os.path.basename(f.filename)
        ext = os.path.splitext(filename)[1].lower()
        prefix = next((p for p in Config.VALID_PREFIXES if filename.lower().startswith(p)), None)
        if ext in Config.ALLOWED_EXTENSIONS and prefix:
            new_filename = f"{prefix}{ext}"
            filepath = os.path.join(folder_path, new_filename)
            f.save(filepath)
            cur.execute("INSERT INTO project_files(project_id, file_type, filename, filepath) VALUES(%s,%s,%s,%s)", (project_id, prefix, filename, filepath))
            
    conn.commit()
    cur.close(); conn.close()
    return jsonify({'message': 'อัปโหลดสำเร็จ'}), 200

@projects_bp.route('/update', methods=['POST'])
@login_required
def update():
    project_id = request.form.get('project_id')
    project_name = request.form.get('project_name', '').strip()
    project_detail = request.form.get('project_detail', '')
    if not project_id or not project_name:
        return jsonify({'error': 'project_id และ project_name ต้องระบุ'}), 400
        
    folder = secure_filename(str(project_id))
    folder_path = os.path.join(Config.UPLOAD_FOLDER, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    try:
        conn = get_db_connection()
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

@projects_bp.route('/delete/<int:project_id>', methods=['POST'])
@login_required
def delete(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return jsonify({'error': 'คุณไม่มีสิทธิ์ลบโปรเจคนี้'}), 403
    cur.close(); conn.close()

    ProjectService.delete_project(project_id)
    return jsonify({'message': 'ลบสำเร็จ'}), 200

@projects_bp.route('/map/<int:project_id>', methods=['GET'])
@login_required
def map_view(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง", 403
    cur.close(); conn.close()

    try:
        path = os.path.join(Config.OUTPUT_FOLDER, str(project_id), "results.json")
        with open(path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        return render_template("testmap.html", project=project_id, result=result_data, user=user)
    except FileNotFoundError:
        return "ยังไม่มีผลลัพธ์การประมวลผล", 404

@projects_bp.route('/run/<int:project_id>', methods=['POST'])
@login_required
def run_project(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return jsonify({'error': 'คุณไม่มีสิทธิ์เรียกใช้งานโปรเจคนี้'}), 403
    cur.close(); conn.close()

    try:
        folder_path = Config.UPLOAD_FOLDER
        result = run_process_from_project_folder(project_id, folder_path)
        if result["success"]:
            return jsonify({"message": "ประมวลผลเสร็จสิ้น"})
        return jsonify({"error": "ประมวลผลไม่สำเร็จ"}), 500
    except Exception as e:
        logging.exception("Error in running project")
        return jsonify({'error': str(e)}), 500

@projects_bp.route('/reprocess/<int:project_id>', methods=['POST'])
@login_required
def reprocess_with_index(project_id):
    try:
        sp_index = request.json.get("sp_index", 0)
        folder_path = Config.UPLOAD_FOLDER
        result = run_process_from_project_folder(project_id, folder_path, sp_index=sp_index)
        if result["success"]:
            return jsonify({"message": "ประมวลผลเสร็จสิ้น"})
        return jsonify({"error": "ประมวลผลไม่สำเร็จ"}), 500
    except Exception as e:
        logging.exception("Error in reprocessing project")
        return jsonify({'error': str(e)}), 500

@projects_bp.route('/download/<int:project_id>')
@login_required
def download_project_files(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return "คุณไม่มีสิทธิ์ดาวน์โหลดโปรเจคนี้", 403
    cur.close(); conn.close()

    folder_path = os.path.join(Config.OUTPUT_FOLDER, str(project_id), 'downloads')
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

@projects_bp.route('/createPeaNoProjects', methods=['POST'])
@login_required
def create_pea_no_project():
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    facility_id = request.form.get("facility_id", "").strip()
    project_detail = request.form.get("project_detail", "").strip()
    region = request.form.get("region", "").strip().upper()
    valid_regions = set(Config.REGION_MAPPING.values())
    if not facility_id:
        return jsonify({"error": "facility_id ต้องระบุ"}), 400
    if region not in valid_regions:
        return jsonify({"error": "ภูมิภาคไม่ถูกต้อง"}), 400
    
    conn = get_db_connection()
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

@projects_bp.route('/runPeaNoProjects/<int:project_id>', methods=['POST'])
@login_required
def run_pea_no_project(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT project_name FROM pea_no_projects WHERE id = %s AND owner_id = %s", (project_id, employee_id))
        project = cur.fetchone()
        if not project: return jsonify({"error": "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง"}), 404
        main_pipeline(project_id=str(project_id), facility_id=project["project_name"])
        return jsonify({"success": True, "message": "ประมวลผลเสร็จสิ้น"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cur.close(); conn.close()

@projects_bp.route('/peaNoMap/<int:project_id>', methods=['GET'])
@login_required
def pea_no_project_map_view(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pea_no_projects WHERE id = %s AND owner_id = %s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง", 403
    cur.close(); conn.close()

    try:
        path = f"pea_no_projects/output/{project_id}/results.json"
        with open(path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        cache_dir = f"pea_no_projects/output/{project_id}/results_cache"
        cached_indices = []
        if os.path.isdir(cache_dir):
            for fname in os.listdir(cache_dir):
                if fname.endswith(".json"):
                    try:
                        cached_indices.append(int(fname[:-5]))
                    except ValueError:
                        pass
            cached_indices.sort()

        recursive_split_path = f"pea_no_projects/output/{project_id}/recursive_split.json"
        if os.path.exists(recursive_split_path):
            with open(recursive_split_path, "r", encoding="utf-8") as f:
                result_data["recursive_split"] = json.load(f)

        return render_template("peaNoProjectmap.html", project=project_id, result=result_data,
                               cached_indices=cached_indices, user=user)
    except FileNotFoundError:
        return "ยังไม่มีผลลัพธ์การประมวลผล", 404

@projects_bp.route('/downloadPeaNoProject/<int:project_id>')
@login_required
def download_pea_no_project_files(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pea_no_projects WHERE id = %s AND owner_id = %s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return "คุณไม่มีสิทธิ์ดาวน์โหลดโปรเจคนี้", 403
    cur.close(); conn.close()

    folder_path = f'pea_no_projects/output/{project_id}/downloads'
    if not os.path.exists(folder_path): return "ไม่พบโฟลเดอร์ดาวน์โหลด", 404
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), folder_path))
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'project_{project_id}_results.zip')

@projects_bp.route('/peaNoProjectResults/<int:project_id>/<int:sp_index>')
@login_required
def pea_no_project_cached_result(project_id, sp_index):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pea_no_projects WHERE id = %s AND owner_id = %s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return jsonify({"error": "ไม่พบโปรเจค"}), 404
    cur.close(); conn.close()
    cache_path = f"pea_no_projects/output/{project_id}/results_cache/{sp_index}.json"
    if not os.path.exists(cache_path):
        return jsonify({"error": "ยังไม่มีผลลัพธ์สำหรับ index นี้"}), 404
    with open(cache_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

@projects_bp.route('/reprocessPeaNoProject/<int:project_id>', methods=['POST'])
@login_required
def pea_no_project_reprocess_with_index(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        body = request.get_json(silent=True) or {}
        sp_index = body.get("sp_index", 0)
        recursive_split = bool(body.get("recursive_split", False))
        max_group_kva = body.get("max_group_kva")
        max_split_depth = body.get("max_split_depth")
        cur.execute("SELECT project_name FROM pea_no_projects WHERE id = %s AND owner_id = %s", (project_id, employee_id))
        project = cur.fetchone()
        if not project: return jsonify({"error": "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง"}), 404
        main_pipeline(project_id=str(project_id), facility_id=project["project_name"], sp_index=sp_index,
                      recursive_split=recursive_split, max_group_kva=max_group_kva, max_split_depth=max_split_depth)
        return jsonify({"success": True, "message": "ประมวลผลเสร็จสิ้น"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cur.close(); conn.close()

@projects_bp.route('/pea_no_project_delete/<int:project_id>', methods=['POST'])
@login_required
def peaNoProjectDelete(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pea_no_projects WHERE id=%s AND owner_id=%s", (project_id, employee_id))
    if not cur.fetchone():
        cur.close(); conn.close()
        return jsonify({'error': 'คุณไม่มีสิทธิ์ลบโปรเจคนี้'}), 403
    cur.close(); conn.close()
    ProjectService.delete_pea_no_project(project_id)
    return jsonify({'message': 'ลบสำเร็จ'}), 200

@projects_bp.route('/pea_no_projects/output/<int:project_id>/<path:filename>')
@login_required
def serve_project_output(project_id, filename):
    base_dir = os.path.join(os.getcwd(), "pea_no_projects", "output", str(project_id))
    return send_from_directory(base_dir, filename)
