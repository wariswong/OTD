import os
import re
import io
import json
import logging
import zipfile
from flask import Blueprint, request, jsonify, render_template, session, send_file, send_from_directory, abort

from ..config import Config
from ..utils.decorators import login_required
from ..services.phase_optimizer_service import PhaseOptimizerService, OUTPUT_ROOT

phase_optimizer_bp = Blueprint('phase_optimizer', __name__)

FAC_RE = re.compile(r'^\d{2}-\d{6}$')


@phase_optimizer_bp.route('/phase_optimizer')
@login_required
def phase_optimizer_list():
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    projects = PhaseOptimizerService.get_projects(employee_id)
    return render_template('phase_optimizer_list.html', projects=projects, user=user)


@phase_optimizer_bp.route('/phase_optimizer/create')
@login_required
def phase_optimizer_create_page():
    user = session.get("user", {})
    return render_template('phase_optimizer_form.html', user=user,
                           regions=sorted(set(Config.REGION_MAPPING.values())))


@phase_optimizer_bp.route('/phase_optimizer/create', methods=['POST'])
@login_required
def phase_optimizer_create():
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    facility_id = request.form.get("facility_id", "").strip()
    project_detail = request.form.get("project_detail", "").strip()
    region = request.form.get("region", "").strip().upper()

    if not FAC_RE.match(facility_id):
        return jsonify({"error": "รูปแบบ FACILITYID ไม่ถูกต้อง (XX-XXXXXX)"}), 400
    if region not in set(Config.REGION_MAPPING.values()):
        return jsonify({"error": "กรุณาเลือกเขต GIS ของหม้อแปลง"}), 400

    project_id = PhaseOptimizerService.create_project(employee_id, facility_id, region, project_detail)
    try:
        PhaseOptimizerService.run(project_id, facility_id, region)
    except Exception as e:
        logging.exception(f"[phase_optimizer] create+run failed for {facility_id}")
        return jsonify({"error": str(e), "project_id": project_id}), 500

    return jsonify({"message": "วิเคราะห์สำเร็จ", "project_id": project_id}), 200


@phase_optimizer_bp.route('/phase_optimizer/run/<int:project_id>', methods=['POST'])
@login_required
def phase_optimizer_run(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    project = PhaseOptimizerService.get_project(project_id, employee_id)
    if not project:
        return jsonify({"error": "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง"}), 404

    try:
        PhaseOptimizerService.run(project_id, project["project_name"], project["region"])
        return jsonify({"message": "วิเคราะห์สำเร็จ"}), 200
    except Exception as e:
        logging.exception(f"[phase_optimizer] run failed for project {project_id}")
        return jsonify({"error": str(e)}), 500


@phase_optimizer_bp.route('/phase_optimizer/delete/<int:project_id>', methods=['POST'])
@login_required
def phase_optimizer_delete(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    project = PhaseOptimizerService.get_project(project_id, employee_id)
    if not project:
        return jsonify({"error": "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง"}), 404

    PhaseOptimizerService.delete_project(project_id)
    return jsonify({"message": "ลบสำเร็จ"}), 200


@phase_optimizer_bp.route('/phase_optimizer/map/<int:project_id>')
@login_required
def phase_optimizer_map_view(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    project = PhaseOptimizerService.get_project(project_id, employee_id)
    if not project:
        return "ไม่พบโปรเจค หรือคุณไม่มีสิทธิ์เข้าถึง", 403

    results_path = os.path.join(OUTPUT_ROOT, str(project_id), "results.json")
    if not os.path.exists(results_path):
        return "ยังไม่มีผลลัพธ์การประมวลผล", 404

    with open(results_path, encoding="utf-8") as fh:
        result_data = json.load(fh)

    return render_template('phase_optimizer_map.html', project=project, project_id=project_id,
                           result=result_data, user=user)


@phase_optimizer_bp.route('/phase_optimizer/download/<int:project_id>')
@login_required
def phase_optimizer_download(project_id):
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    project = PhaseOptimizerService.get_project(project_id, employee_id)
    if not project:
        return "คุณไม่มีสิทธิ์ดาวน์โหลดโปรเจคนี้", 403

    folder_path = os.path.join(OUTPUT_ROOT, str(project_id), "downloads")
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
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True,
                     download_name=f'phase_optimizer_{project_id}_results.zip')


@phase_optimizer_bp.route('/phase_optimizer/output/<int:project_id>/<path:filename>')
@login_required
def phase_optimizer_serve_output(project_id, filename):
    # Geometry files aren't sensitive per-user secrets, but keep the same
    # ownership check as the other routes rather than serving them to anyone
    # who guesses a project_id.
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    project = PhaseOptimizerService.get_project(project_id, employee_id)
    if not project:
        abort(403)
    base_dir = os.path.join(OUTPUT_ROOT, str(project_id))
    return send_from_directory(base_dir, filename)
