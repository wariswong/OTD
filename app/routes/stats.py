import logging
import traceback
from flask import Blueprint, request, render_template, jsonify, session, redirect, url_for
from ..utils.decorators import login_required, is_admin
from ..utils.helpers import get_user_region
from ..services.stats_service import StatsService
from ..services.admin_service import AdminService
from ..database import get_db_connection
from ..config import Config

stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/transformer_stats')
@login_required
def transformer_stats():
    try:
        region = get_user_region()
        page = request.args.get('page', 1, type=int)
        search_query = request.args.get('q', '').strip()
        filter_problem = request.args.get('filter_problem', '').strip()
        filter_fix     = request.args.get('filter_fix', '').strip()
        filter_name    = request.args.get('filter_name', '').strip()
        per_page = 15
        offset = (page - 1) * per_page

        stats, total_rows = StatsService.get_transformer_stats(
            region, search_query, per_page, offset,
            filter_problem=filter_problem or None,
            filter_fix=filter_fix or None,
            filter_name=filter_name or None,
        )
        total_pages = (total_rows + per_page - 1) // per_page
        problem_options, fix_options, name_options = StatsService.get_filter_options(region)

        user = session.get("user", {})
        return render_template('transformer_stats.html',
                             stats=stats,
                             page=page,
                             total_pages=total_pages,
                             total_rows=total_rows,
                             region=region,
                             search_query=search_query,
                             filter_problem=filter_problem,
                             filter_fix=filter_fix,
                             filter_name=filter_name,
                             problem_options=problem_options,
                             fix_options=fix_options,
                             name_options=name_options,
                             user=user,
                             user_department=user.get('hr_department', ''))
    except Exception as e:
        logging.error(f"Error in transformer_stats: {e}\n{traceback.format_exc()}")
        return "เกิดข้อผิดพลาดในระบบ กรุณาติดต่อผู้ดูแล", 500

@stats_bp.route('/transformer_stats/map_data')
@login_required
def transformer_stats_map_data():
    try:
        region = get_user_region()
        search_query   = request.args.get('q', '').strip()
        filter_problem = request.args.get('filter_problem', '').strip()
        filter_fix     = request.args.get('filter_fix', '').strip()
        filter_name    = request.args.get('filter_name', '').strip()

        stats, _ = StatsService.get_transformer_stats(
            region, search_query, per_page=9999, offset=0,
            filter_problem=filter_problem or None,
            filter_fix=filter_fix or None,
            filter_name=filter_name or None,
        )
        data = [
            {
                'facility_id': r.get('facility_id'),
                'location': r.get('location') or '',
                'lat': float(r['lat']) if r.get('lat') is not None else None,
                'lon': float(r['lon']) if r.get('lon') is not None else None,
                'rate_kva': r.get('rate_kva'),
                'pload_peak': r.get('pload_peak'),
                'problem_summary': r.get('problem_summary') or '',
                'fix_guideline': r.get('fix_guideline') or '',
                'name': r.get('name') or '',
                'x': r.get('x'),
                'y': r.get('y'),
            }
            for r in stats
            if r.get('lat') is not None and r.get('lon') is not None
        ]
        return jsonify(data)
    except Exception as e:
        logging.error(f"map_data error: {e}\n{traceback.format_exc()}")
        return jsonify([]), 500

@stats_bp.route('/admin/upload_stats', methods=['GET', 'POST'])
@login_required
def admin_upload_stats():
    if not is_admin(): return "Access Denied", 403
    
    role = session.get("admin_role")
    admin_region = session.get("admin_region") # e.g. "A", "B", "C"
    
    if request.method == 'POST':
        file = request.files.get('file')
        target_region = request.form.get('region') # e.g. "N1", "N2"
        overwrite = request.form.get('overwrite') == 'true'
        
        # Validation: Region Admin can only upload for their own region mapping
        if role == 'admin_region':
            valid_target = False
            for letter, region_name in Config.REGION_MAPPING.items():
                if letter == admin_region and region_name == target_region:
                    valid_target = True
                    break
            
            if not valid_target:
                return jsonify({"error": f"คุณมีสิทธิ์อัปโหลดเฉพาะเขต {admin_region} ({Config.REGION_MAPPING.get(admin_region)}) เท่านั้น"}), 403

        if file and target_region:
            try:
                count = StatsService.upload_stats(file.stream, target_region, overwrite)
                return jsonify({"message": f"Successfully uploaded {count} rows for region {target_region}"})
            except Exception as e:
                return jsonify({"error": f"Upload error: {str(e)}"}), 500
                
    return render_template('upload_stats.html', role=role, admin_region=admin_region)

@stats_bp.route('/admin/management', methods=['GET', 'POST'])
@login_required
def admin_management():
    if session.get("admin_role") != 'admin_system':
        return "Access Denied: System Admin only", 403
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add':
            eid = request.form.get('hr_employee_id')
            role = request.form.get('role')
            region = request.form.get('region') if role == 'admin_region' else None
            AdminService.add_admin(eid, role, region)
        elif action == 'delete':
            admin_id = request.form.get('admin_id')
            AdminService.delete_admin(admin_id)
        return redirect(url_for('stats.admin_management'))
        
    admins = AdminService.get_all_admins()
    return render_template('admin_management.html', admins=admins, user=session.get("user"))

@stats_bp.route('/admin/migrate_stats')
@login_required
def admin_migrate_stats():
    if not is_admin(): return "Access Denied", 403
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        with open('db/create_stats_table.sql', 'r', encoding='utf-8') as f:
            sql = f.read()
        results = cur.execute(sql, multi=True)
        for res in results:
            pass
        conn.commit(); cur.close(); conn.close()
        return "Migration Successful"
    except Exception as e: 
        return f"Migration Failed: {str(e)}"
