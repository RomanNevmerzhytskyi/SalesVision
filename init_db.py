# init_db.py
from app import app, db, user_datastore
from flask_security.utils import hash_password

with app.app_context():
    db.drop_all()
    db.create_all()
    user_datastore.create_user(email="admin@salesvision.com", password=hash_password("123456"))
    db.session.commit()
    print("✅ Database initialized successfully.")
