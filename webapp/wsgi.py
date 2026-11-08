"""Production entry point:  gunicorn webapp.wsgi:app

On a fresh server there is no data/app.db (it is git-ignored), so it is rebuilt from the committed SQL dump
(data/dump/app_dump.sql), which keeps the same logins as the credential files. The Part A database
(data/instamart.db) is built at deploy time with `python -m src.data_pipeline`.
"""
from webapp import records
from webapp.app import create_app
from webapp.config import Config

cfg = Config()
if not cfg.app_db.exists():
    records.restore_app(cfg)
app = create_app(cfg)
