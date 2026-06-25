import os
from flask import session
from ..config import Config

def get_user_region():
    user = session.get("user", {})
    business_area = user.get("hr_business_area", "")
    if business_area:
        char = business_area[0].upper()
        return Config.REGION_MAPPING.get(char, "NE2")
    return "NE2"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
