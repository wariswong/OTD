import os
import re
from pathlib import Path

from ..database import get_db_connection

# feature_shareload/FEASIBLE/<facA>_<facB>/ — shareload has no DB table (see
# app/routes/shareload.py's _list_pairs()); each folder is one completed job.
# No owner_id is ever recorded for a shareload run, so it can only be counted,
# not attributed to a user.
_ROOT = Path(os.path.dirname(__file__)).parent.parent
_FEASIBLE_DIR = _ROOT / "feature_shareload" / "FEASIBLE"
_PAIR_RE = re.compile(r"^\d{2}-\d{6}_\d{2}-\d{6}$")

# (label shown in the report, table name, project_name column's display name)
_JOB_TABLES = [
    ("ตั้งเสริมตัดจ่าย", "pea_no_projects"),
    ("อัปโหลด Shapefile", "projects"),
    ("ออกแบบงานปรับปรุงทั่วไป (Phase Optimizer)", "phase_optimizer_projects"),
]


class AdminReportService:
    @staticmethod
    def get_login_stats():
        """(total_logins, distinct_users) from login_log. Table may not exist
        yet on a fresh DB — treat that the same as "no data" rather than 500.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT hr_employee_id) FROM login_log")
            total, distinct_users = cur.fetchone()
            return total or 0, distinct_users or 0
        except Exception:
            return 0, 0
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_top_users(limit=10):
        """Most frequent visitors — (hr_employee_id, hr_fullname_th, login_count, last_login)."""
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT hr_employee_id,
                       MAX(hr_fullname_th) AS hr_fullname_th,
                       MAX(hr_department) AS hr_department,
                       COUNT(*) AS login_count,
                       MAX(logged_in_at) AS last_login
                FROM login_log
                WHERE hr_employee_id IS NOT NULL
                GROUP BY hr_employee_id
                ORDER BY login_count DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()
        except Exception:
            return []
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def _shareload_job_count():
        if not _FEASIBLE_DIR.exists():
            return 0
        return sum(1 for f in _FEASIBLE_DIR.iterdir() if f.is_dir() and _PAIR_RE.match(f.name))

    @staticmethod
    def get_job_counts():
        """[{label, count}, ...] — one row per feature, in the same order as _JOB_TABLES,
        plus shareload (folder-based, no DB table)."""
        conn = get_db_connection()
        cur = conn.cursor()
        counts = []
        try:
            for label, table in _JOB_TABLES:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    counts.append({"label": label, "count": cur.fetchone()[0]})
                except Exception:
                    counts.append({"label": label, "count": 0})
        finally:
            cur.close()
            conn.close()
        counts.append({
            "label": "วิเคราะห์การแบ่งโหลดหม้อแปลงข้างเคียง (Shareload)",
            "count": AdminReportService._shareload_job_count(),
        })
        return counts

    @staticmethod
    def get_recent_jobs(limit=30):
        """Most recent jobs across every DB-backed feature, newest first.
        Shareload isn't included here (folder mtime isn't a reliable "created_at"
        and it has no owner) — it's shown separately as a total count only.
        """
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        rows = []
        try:
            for label, table in _JOB_TABLES:
                try:
                    cur.execute(
                        f"SELECT project_name, project_detail, owner_id, "
                        f"{'region, ' if table != 'projects' else 'NULL AS region, '}"
                        f"created_at FROM {table} ORDER BY created_at DESC LIMIT %s",
                        (limit,),
                    )
                    for r in cur.fetchall():
                        r["job_type"] = label
                        rows.append(r)
                except Exception:
                    continue
        finally:
            cur.close()
            conn.close()
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        rows = rows[:limit]

        owner_ids = {r["owner_id"] for r in rows if r.get("owner_id")}
        names = AdminReportService._names_for(owner_ids)
        for r in rows:
            r["owner_name"] = names.get(r.get("owner_id"))
        return rows

    @staticmethod
    def _names_for(employee_ids):
        """Best-effort hr_employee_id -> hr_fullname_th lookup from login_log
        (the closest thing to a users table this app has)."""
        if not employee_ids:
            return {}
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            placeholders = ",".join(["%s"] * len(employee_ids))
            cur.execute(
                f"SELECT hr_employee_id, hr_fullname_th FROM login_log "
                f"WHERE hr_employee_id IN ({placeholders}) "
                f"ORDER BY logged_in_at DESC",
                tuple(employee_ids),
            )
            names = {}
            for eid, name in cur.fetchall():
                names.setdefault(eid, name)  # first hit = most recent (DESC order)
            return names
        except Exception:
            return {}
        finally:
            cur.close()
            conn.close()
