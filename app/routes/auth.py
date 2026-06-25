import logging
from flask import Blueprint, request, jsonify, redirect, url_for, session
import requests
from ..config import Config

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/login")
def login():
    params = {
        "redirect_uri": Config.REDIRECT_URI_CALLBACK,
        "response_type": "code",
        "scope": "openid profile",
        "client_id": Config.CLIENT_ID
    }
    encoded_params = requests.compat.urlencode(params)
    return redirect(f"{Config.SSO_AUTH_URL}?{encoded_params}")

@auth_bp.route("/login/callback")
def login_callback():
    code = request.args.get("code")
    if not code:
        return "No code provided", 400
    
    data = {
        "code": code,
        "client_id": Config.CLIENT_ID,
        "client_secret": Config.CLIENT_SECRET,
        "redirect_uri": Config.REDIRECT_URI_CALLBACK,
        "grant_type": "authorization_code",
    }
    
    try:
        r = requests.post(
            Config.SSO_TOKEN_URL, 
            data=data, 
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        r.raise_for_status()
        token_json = r.json()
        access_token = token_json.get("access_token")
        
        if not access_token:
            return "Failed to get access token", 400
            
        u = requests.get(
            Config.SSO_USERINFO_URL, 
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )
        u.raise_for_status()
        userinfo = u.json()
        session["user"] = userinfo
        return redirect(url_for("stats.transformer_stats"))
    except requests.exceptions.RequestException as e:
        logging.error(f"SSO login error: {e}")
        return "เกิดข้อผิดพลาดในการเข้าสู่ระบบ SSO กรุณาลองใหม่อีกครั้ง", 500

@auth_bp.route("/logout")
def logout():
    session.clear()
    params = {
        "client_id": Config.CLIENT_ID,
        "post_logout_redirect_uri": Config.REDIRECT_URI,
        "lockout": "true"
    }
    encoded_params = requests.compat.urlencode(params)
    return redirect(f"{Config.SSO_LOGOUT_URL}?{encoded_params}")

@auth_bp.route("/api/userinfo")
def api_userinfo():
    return jsonify(session.get("user", {}))
