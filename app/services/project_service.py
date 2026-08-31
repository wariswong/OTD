import os
import shutil
import logging
import json
import zipfile
import io
from werkzeug.utils import secure_filename
from ..config import Config
from ..database import get_db_connection
from ..utils.helpers import get_user_region

# Existing logic imports (kept as is for compatibility)
from processNew_no_gui import run_process_from_project_folder
from InputJsonApi import run_pipeline_for_facilityid
from optimized_transformer_group_310869 import main_pipeline

class ProjectService:
    @staticmethod
    def get_shape_projects(employee_id):
        conn = get_db_connection()
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
            output_path = os.path.join(Config.OUTPUT_FOLDER, str(project["id"]))
            project["has_output"] = os.path.exists(output_path) and len(os.listdir(output_path)) > 0
        return projects

    @staticmethod
    def get_pea_no_projects(employee_id):
        conn = get_db_connection()
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
            output_path = os.path.join("pea_no_projects", "output", str(project["id"]))
            project["has_output"] = os.path.exists(output_path) and len(os.listdir(output_path)) > 0
        return projects

    @staticmethod
    def delete_project(project_id):
        folder = secure_filename(str(project_id))
        upload_folder_path = os.path.join(Config.UPLOAD_FOLDER, folder)
        output_folder_path = os.path.join(Config.OUTPUT_FOLDER, folder)
        
        conn = get_db_connection()
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
        return True

    @staticmethod
    def delete_pea_no_project(project_id):
        folder = secure_filename(str(project_id))
        input_folder_path = os.path.join("pea_no_projects", "input", folder)
        output_folder_path = os.path.join("pea_no_projects", "output", folder)
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM pea_no_projects WHERE id=%s", (project_id,))
        conn.commit()
        cur.close(); conn.close()
        
        try:
            if os.path.isdir(input_folder_path): shutil.rmtree(input_folder_path)
            if os.path.isdir(output_folder_path): shutil.rmtree(output_folder_path)
        except Exception as e:
            logging.error(f"Delete folder error: {e}")
        return True
