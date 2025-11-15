from flask_security.forms import LoginForm
from wtforms import BooleanField

class ExtendedLoginForm(LoginForm):
    remember = BooleanField("Remember Me")
