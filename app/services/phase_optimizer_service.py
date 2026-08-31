import os
import shutil
import logging
from ..database import get_db_connection

# feature_PhaseOptimizer/run_web.py has a hyphen-free name but lives outside
# the app/ package — load it the same way feature_shareload/run_web.py's
# TransferOptimizer-072026.py dependency gets loaded, by absolute file path,
# so it resolves regardless of the server's cwd.
import importlib.util
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_ROOT = os.path.join(_ROOT, "feature_PhaseOptimizer", "output")

_spec = importlib.util.spec_from_file_location(
    "phase_optimizer_run_web",
    os.path.join(_ROOT, "feature_PhaseOptimizer", "run_web.py"),
)
_run_web_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_web_mod)
run_phase_optimizer = _run_web_mod.run_phase_optimizer


class PhaseOptimizerService:
    @staticmethod
    def get_projects(employee_id):
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT p.*
            FROM phase_optimizer_projects p
            WHERE p.owner_id = %s
            ORDER BY p.created_at DESC
        """, (employee_id,))
        projects = cur.fetchall()
        cur.close(); conn.close()

        for project in projects:
            output_path = os.path.join(OUTPUT_ROOT, str(project["id"]))
            project["has_output"] = os.path.exists(os.path.join(output_path, "results.json"))
        return projects

    @staticmethod
    def create_project(employee_id, facility_id, region, project_detail=""):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO phase_optimizer_projects (project_name, project_detail, owner_id, region, created_at) "
            "VALUES (%s, %s, %s, %s, NOW())",
            (facility_id, project_detail, employee_id, region),
        )
        conn.commit()
        project_id = cur.lastrowid
        cur.close(); conn.close()
        return project_id

    @staticmethod
    def get_project(project_id, employee_id):
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM phase_optimizer_projects WHERE id=%s AND owner_id=%s",
            (project_id, employee_id),
        )
        project = cur.fetchone()
        cur.close(); conn.close()
        return project

    @staticmethod
    def run(project_id, facility_id, region):
        """Run the optimizer synchronously and return the results dict.

        Raises on failure — callers should catch and surface the message,
        same as optimized_transformer_group_310869.main_pipeline's callers do.
        """
        out_dir = os.path.join(OUTPUT_ROOT, str(project_id))
        return run_phase_optimizer(facility_id, out_dir, region=region)

    @staticmethod
    def delete_project(project_id):
        output_path = os.path.join(OUTPUT_ROOT, str(project_id))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM phase_optimizer_projects WHERE id=%s", (project_id,))
        conn.commit()
        cur.close(); conn.close()

        try:
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
        except Exception as e:
            logging.error(f"[phase_optimizer] delete output folder error: {e}")
        return True
