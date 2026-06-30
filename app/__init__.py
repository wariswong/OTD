import os
from flask import Flask
from .config import Config

def create_app(config_class=Config):
    # Get the absolute path to the project root (OTD/)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    app = Flask(__name__,
                root_path=root_dir,
                template_folder="templates",
                static_folder="output",
                static_url_path="/output")
                
    app.config.from_object(config_class)

    # Register Blueprints
    from .routes.auth import auth_bp
    from .routes.projects import projects_bp
    from .routes.stats import stats_bp
    from .routes.shareload import shareload_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(shareload_bp)

    @app.context_processor
    def utility_processor():
        from flask import request
        from .utils.decorators import is_admin
        return dict(request=request, is_admin=is_admin)

    # Optional: Serve the 'static' folder if it's used in templates
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        from flask import send_from_directory
        return send_from_directory(os.path.join(app.root_path, 'static'), filename)

    return app
