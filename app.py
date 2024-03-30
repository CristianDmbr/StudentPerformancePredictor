from flask import Flask, render_template, url_for, redirect, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

app = Flask(__name__, static_url_path='/static')
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'sessionkey'

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegistrationForm(FlaskForm): 
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

    def validate_username(self, username): 
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            flash("That username is already taken. Please choose a different one.", "error")
            raise ValidationError("That username already exists. Please choose a different one.")

class LoginForm(FlaskForm): 
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    form = LoginForm()
    error = None 

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid password. Please try again."
        else:
            error = "Username doesn't exist. Please sign up."

    return render_template('login.html', form=form, error=error)

@app.route('/dashboard')
@login_required
def dashboard():
    academic_programs = ['INDUSTRIAL ENGINEERING', 'ELECTRONIC ENGINEERING', 'CIVIL ENGINEERING',
                     'MECHANICAL ENGINEERING', 'ELECTRIC ENGINEERING',
                     'ELECTRIC ENGINEERING AND TELECOMMUNICATIONS', 'CHEMICAL ENGINEERING',
                     'AERONAUTICAL ENGINEERING', 'MECHATRONICS ENGINEERING',
                     'INDUSTRIAL AUTOMATIC ENGINEERING', 'TRANSPORTATION AND ROAD ENGINEERING',
                     'TOPOGRAPHIC ENGINEERY', 'INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING',
                     'CONTROL ENGINEERING', 'CATASTRAL ENGINEERING AND GEODESY',
                     'PRODUCTION ENGINEERING', 'PRODUCTIVITY AND QUALITY ENGINEERING',
                     'CIVIL CONSTRUCTIONS', 'ELECTROMECHANICAL ENGINEERING',
                     'AUTOMATION ENGINEERING', 'TEXTILE ENGINEERING']

    performance_saber11 = ['Good', 'Very Good', 'Fail', 'Pass', 'Best']

    return render_template('dashboard.html', academic_programs=academic_programs, performance_saber11=performance_saber11)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=["GET", "POST"])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if not existing_user:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
            new_user = User(username=form.username.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login')) 
        else:
            flash("That username is already taken. Please choose a different one.", "error")

    return render_template('register.html', form=form)

@app.route('/parent-info', methods=['GET', 'POST'])
@login_required
def parent_info():
    if request.method == 'POST':
        next_page = request.form.get('next', None)
        if next_page == 'socioeconomic_info':
            return redirect(url_for('socioeconomic_info'))
    return render_template('parent_info.html')

@app.route('/socioeconomic-info', methods=['GET', 'POST'])
@login_required
def socioeconomic_info():
    if request.method == 'POST':
        next_page = request.form.get('next', None)
        if next_page == 'high_school_info':
            return redirect(url_for('high_school_info'))
    return render_template('socioeconomic_info.html')

@app.route('/high-school-info', methods=['GET', 'POST'])
@login_required
def high_school_info():
    if request.method == 'POST':
        next_page = request.form.get('next', None)
        if next_page == 'university_info':
            return redirect(url_for('university_info'))
    return render_template('high_school_info.html')

@app.route('/university-info', methods=['GET', 'POST'])
@login_required
def university_info():
    if request.method == 'POST':
        next_page = request.form.get('next', None)
        if next_page == 'result':
            return redirect(url_for('result'))
    return render_template('university_info.html')

@app.route('/result', methods=['GET', 'POST'])  # Allow both GET and POST methods
@login_required
def result():
    if request.method == 'POST':
        # Handle any form submissions if necessary
        pass  # Placeholder, you can add handling logic here
    
    return render_template('result.html')

if __name__ == '__main__':
    with app.app_context():
        inspector = db.inspect(db.engine)
        if 'user' in inspector.get_table_names():
            print("User table exists.")
        else:
            print("User table does not exist.")
        
        db.create_all()
    
    app.run(debug=True)