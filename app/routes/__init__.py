def register_blueprints(app):
    from app.routes.health import health_bp
    from app.routes.brief import brief_bp
    from app.routes.feedback import feedback_bp
    from app.routes.admin import admin_bp
    from app.routes.cost import cost_bp
    from app.routes.views import views_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(brief_bp, url_prefix='/api/brief')
    app.register_blueprint(feedback_bp, url_prefix='/api/feedback')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(cost_bp, url_prefix='/api/cost')
    app.register_blueprint(views_bp)

    # Telegram bot (conditional — only if token is configured)
    if app.config.get('TELEGRAM_BOT_TOKEN'):
        from app.routes.telegram import telegram_bp
        app.register_blueprint(telegram_bp, url_prefix='/api/telegram')
