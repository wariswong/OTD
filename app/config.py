import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key-please-change")
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    
    # DB Config
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "odt")

    # SSO Config
    SSO_AUTH_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/auth"
    SSO_TOKEN_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/token"
    SSO_USERINFO_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/userinfo"
    SSO_LOGOUT_URL = "https://sso2.pea.co.th/realms/pea-users/protocol/openid-connect/logout"

    CLIENT_ID = os.getenv("SSO_CLIENT_ID")
    CLIENT_SECRET = os.getenv("SSO_CLIENT_SECRET")
    REDIRECT_URI = os.getenv("SSO_REDIRECT_URI")
    REDIRECT_URI_CALLBACK = os.getenv("SSO_REDIRECT_URI_CALLBACK")

    REGION_MAPPING = {
        'A': 'N1', 'B': 'N2', 'C': 'N3',
        'D': 'NE1', 'E': 'NE2', 'F': 'NE3',
        'G': 'C1', 'H': 'C2', 'I': 'C3',
        'J': 'S1', 'K': 'S2', 'L': 'S3',
        'Z': 'Z'
    }

    ALLOWED_EXTENSIONS = {'shp', 'shx', 'dbf', 'prj', 'cpg', 'sbn', 'sbx'}
    VALID_PREFIXES = ['meter', 'lv', 'mv', 'tr', 'eservice']
    ADMIN_IDS = os.getenv("ADMIN_IDS", "").split(",") if os.getenv("ADMIN_IDS") else []
