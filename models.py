# models.py
from extensions import db
from flask_security import UserMixin, RoleMixin  # ✅ add this
from datetime import datetime
import uuid

class Role(db.Model, RoleMixin):  # ✅ RoleMixin gives proper role methods
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

class User(db.Model, UserMixin):  # ✅ UserMixin fixes your error
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean(), default=True)
    confirmed_at = db.Column(db.DateTime())
    fs_uniquifier = db.Column(
        db.String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    roles = db.relationship('Role', secondary=roles_users, backref=db.backref('users', lazy='dynamic'))
    uploads = db.relationship('Upload', back_populates='user', lazy=True)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(255))
    
    user = db.relationship('User', back_populates='uploads')
    

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    theme = db.Column(db.String(20), default='light')                 # 'light' | 'dark'
    charts_per_row = db.Column(db.Integer, default=2)                 # 1, 2, 3...
    homepage_bg = db.Column(db.String(9), default='#f1f3f6')          # hex like #f1f3f6
    default_dashboard_tab = db.Column(db.String(32), default='overview')

    user = db.relationship('User', backref=db.backref('preference', uselist=False))
