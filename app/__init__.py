import logging
import os
from flask import Flask
from config import Config


def create_app(config_class=None):
    app = Flask(__name__)
    app.config.from_object(config_class or Config)

    # Fix Railway's DATABASE_URL (postgres:// -> postgresql://)
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    if db_uri.startswith('postgres://'):
        app.config['SQLALCHEMY_DATABASE_URI'] = db_uri.replace('postgres://', 'postgresql://', 1)

    # Logging
    logging.basicConfig(
        level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Extensions
    from app.extensions import db, migrate, scheduler
    db.init_app(app)
    migrate.init_app(app, db)

    # Feature flags
    from app import feature_flags
    feature_flags.init_flags()

    # Register blueprints
    from app.routes import register_blueprints
    register_blueprints(app)

    # Scheduler
    if app.config.get('SCHEDULER_ENABLED') and not app.config.get('TESTING'):
        scheduler.init_app(app)
        with app.app_context():
            from app.jobs.scheduled import register_jobs
            register_jobs(scheduler, app)
        scheduler.start()

    return app
