from functools import wraps
from flask import session, redirect, url_for
from ..config import Config
from ..services.admin_service import AdminService

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function

def is_admin():
    """Checks if current user is an admin (of any type)."""
    user = session.get("user", {})
    employee_id = user.get("hr_employee_id")
    if not employee_id: return False
    
    admin_data = AdminService.get_admin_by_id(employee_id)
    if admin_data:
        # Cache role and region in session for convenience
        session["admin_role"] = admin_data["role"]
        session["admin_region"] = admin_data["region"]
        return True
    return False

def get_admin_role():
    """Returns 'admin_system', 'admin_region' or None."""
    if is_admin():
        return session.get("admin_role")
    return None
