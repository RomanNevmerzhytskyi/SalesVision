# models.py
from extensions import db
from flask_security import UserMixin, RoleMixin
from datetime import datetime
import uuid


class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))


roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean(), default=True)
    confirmed_at = db.Column(db.DateTime())
    fs_uniquifier = db.Column(
        db.String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )

    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))
    uploads = db.relationship('Upload', back_populates='user', lazy=True)

    # keep access to all rows across all uploads
    sales_records = db.relationship('SalesRecord', back_populates='user', lazy=True)


class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(255))

    user = db.relationship('User', back_populates='uploads')

    # ⬇️ important: don't cascade delete children, just let upload go away
    records = db.relationship(
        'SalesRecord',
        back_populates='upload',
        lazy=True,
        passive_deletes=True,       # don't issue UPDATEs on child rows when parent is deleted
    )


class SalesRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # ⬇️ make this nullable so rows can survive after Upload is deleted
    upload_id = db.Column(
        db.Integer,
        db.ForeignKey('upload.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
    )
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)

    row_hash = db.Column(db.String(64), unique=True, index=True, nullable=False)

    date     = db.Column(db.Date)
    product  = db.Column(db.String(255))
    category = db.Column(db.String(255))
    region   = db.Column(db.String(255))
    customer = db.Column(db.String(255))

    quantity   = db.Column(db.Float)
    unit_price = db.Column(db.Float)
    sales      = db.Column(db.Float)
    profit     = db.Column(db.Float)

    upload = db.relationship('Upload', back_populates='records')
    user   = db.relationship('User', back_populates='sales_records')


class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    theme = db.Column(db.String(20), default='light')
    charts_per_row = db.Column(db.Integer, default=2)
    homepage_bg = db.Column(db.String(9), default='#f1f3f6')
    default_dashboard_tab = db.Column(db.String(32), default='overview')

    user = db.relationship('User', backref=db.backref('preference', uselist=False))
