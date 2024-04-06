from flask import Flask, render_template, url_for, redirect, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from heapq import nlargest, nsmallest
import joblib 

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
    return render_template('dashboard.html', username=current_user.username) 

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
    performance_saber11 = ['Good', 'Very Good', 'Fail', 'Pass', 'Best']

    if request.method == 'POST':
        next_page = request.form.get('next', None)
        if next_page == 'university_info':
            return redirect(url_for('university_info'))
    return render_template('high_school_info.html', performance_saber11=performance_saber11)


@app.route('/university-info', methods=['GET', 'POST'])
@login_required
def university_info():
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

        if request.method == 'POST':
            next_page = request.form.get('next', None)
            if next_page == 'result':
                return redirect(url_for('result'))
        return render_template('university_info.html', academic_programs=academic_programs)


# Load data and train Random Forest model
dataBase = pd.read_csv('Database/Academic Database.csv')
dataBase.drop(columns=['COD_S11', 'COD_SPRO', 'SCHOOL_NAME', 'UNIVERSITY', 'Unnamed: 9'], inplace=True)

# Adding new columns:
dataBase["Average_Score_SABER_PRO"] = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']].mean(axis=1)

# Define performance levels based on average score
dataBase["Performance_SABER_PRO"] = np.select(
    [
        dataBase['Average_Score_SABER_PRO'] >= 90,
        dataBase['Average_Score_SABER_PRO'] >= 80,
        dataBase['Average_Score_SABER_PRO'] >= 70,
        dataBase['Average_Score_SABER_PRO'] >= 60,
        dataBase['Average_Score_SABER_PRO'] >= 50
    ],
    [
        'Best',
        'Very Good',
        'Good',
        'Below Average performance',
        'Student will pass'
    ],
    default='Student is under risk of failure, urgent help is needed'
)

dataBase["Average_Score_Saber11"] = dataBase[['MAT_S11', 'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11']].mean(axis=1)
dataBase["Performance_Saber11"] = np.select(
    [
        dataBase['Average_Score_Saber11'] >= 90,
        dataBase['Average_Score_Saber11'] >= 80,
        dataBase['Average_Score_Saber11'] >= 70,
        dataBase['Average_Score_Saber11'] >= 60
    ],
    [
        'Best',
        'Very Good',
        'Good',
        'Pass'
    ],
    default='Urgent academic assistance is needed'
)

dataBase["EDU_FATHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
dataBase["EDU_MOTHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
dataBase["OCC_FATHER"].replace({"0": "Unemployed", "Home" : "Unemployed", "Retired": "Unemployed"}, inplace=True)
dataBase["OCC_MOTHER"].replace({"0": "Unemployed", "Home" : "Unemployed", "Retired": "Unemployed"}, inplace=True)
dataBase["SISBEN"].replace({"Esta clasificada en otro Level del SISBEN" : "It is classified in another SISBEN Level"}, inplace=True)
dataBase["PEOPLE_HOUSE"].replace({"Nueve" : "Nine", "Once" : "Eleven"}, inplace=True)
dataBase["JOB"].replace({"No":"0 hours per week","0":"0 hours per week"}, inplace=True)

X_raw = dataBase[['GENDER', 'EDU_FATHER', 'EDU_MOTHER', 'OCC_FATHER', 'OCC_MOTHER',
       'STRATUM', 'SISBEN', 'PEOPLE_HOUSE', 'INTERNET', 'TV',
       'COMPUTER', 'WASHING_MCH', 'MIC_OVEN', 'CAR', 'DVD', 'FRESH', 'PHONE',
       'MOBILE', 'REVENUE', 'JOB', 'SCHOOL_NAT', 'SCHOOL_TYPE', 'MAT_S11',
       'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11', 'ACADEMIC_PROGRAM', 'QR_PRO',
       'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO', 'FEP_PRO', 'G_SC',
       'PERCENTILE', '2ND_DECILE', 'QUARTILE', 'SEL', 'SEL_IHE',
       'Average_Score_Saber11', 'Performance_Saber11']]

y_raw = dataBase["Performance_SABER_PRO"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

missing_columns = X_train_raw.columns[X_train_raw.isnull().any()]
numerical_features = X_raw.select_dtypes(include=np.number).columns
categorical_features = X_raw.select_dtypes(exclude=np.number).columns

if not X_train_raw[numerical_features].empty:
    numeric_imputer = SimpleImputer(strategy="mean")
    X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features])
    X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features])
else:
    print("No numerical features to impute.")

if not X_train_raw[categorical_features].empty:
    categoric_imputer = SimpleImputer(strategy="most_frequent")
    X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features])
    X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features])
else:
    print("No categorical features to impute.")

encoder = OneHotEncoder(handle_unknown='ignore')
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed)
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed)

X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded.toarray()), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded.toarray()), axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

clf_rf = RandomForestClassifier(bootstrap = False,
                                max_depth = 20,
                                min_samples_leaf = 1,
                                min_samples_split = 5,
                                n_estimators = 50)
clf_rf.fit(X_train_scaled, y_train)

joblib.dump(clf_rf, 'random_forest_model.joblib')


@app.route('/result', methods=['POST'])
@login_required
def result():
    if request.method == 'POST':
        # Retrieve input values from the university information form
        academic_program = request.form.get('academic_program')
        qr_pro = float(request.form.get('qr_pro'))
        cr_pro = float(request.form.get('cr_pro'))
        cc_pro = float(request.form.get('cc_pro'))
        eng_pro = float(request.form.get('eng_pro'))
        wc_pro = float(request.form.get('wc_pro'))
        fep_pro = float(request.form.get('fep_pro'))
        g_sc = float(request.form.get('g_sc'))
        percentile = float(request.form.get('percentile'))
        second_decile = float(request.form.get('2nd_decile'))
        quartile = float(request.form.get('quartile'))
        sel = float(request.form.get('sel'))
        sel_ihe = float(request.form.get('sel_ihe'))

        # Retrieve input values from the high school information form
        school_nat = request.form.get('school_nat')
        school_type = request.form.get('school_type')
        mat_s11 = request.form.get('mat_s11')
        cr_s11 = request.form.get('cr_s11')
        cc_s11 = request.form.get('cc_s11')
        bio_s11 = request.form.get('bio_s11')
        eng_s11 = request.form.get('eng_s11')
        average_score_saber11 = request.form.get('average_score_saber11')
        performance_saber11 = request.form.get('performance_saber11')

        # Retrieve input values from the socioeconomic information form
        stratum = request.form.get('stratum')
        sisben = request.form.get('sisben')
        people_house = request.form.get('people_house')  # Retrieve the value without converting it to int yet
        internet = request.form.get('internet')
        tv = request.form.get('tv')
        computer = request.form.get('computer')
        washing_machine = request.form.get('washing_machine')
        microwave_oven = request.form.get('microwave_oven')
        car = request.form.get('car')
        dvd = request.form.get('dvd')
        phone = request.form.get('phone')
        mobile = request.form.get('mobile')
        fresh_food = request.form.get('fresh_food')
        family_revenue = request.form.get('family_revenue')
        job = request.form.get('job')

        # Retrieve input values from the parent information form
        gender = request.form.get('gender')
        edu_father = request.form.get('edu_father')
        edu_mother = request.form.get('edu_mother')
        occ_father = request.form.get('occ_father')
        occ_mother = request.form.get('occ_mother')

        # Create a DataFrame containing all input values
        input_data = pd.DataFrame({
            'GENDER': [gender],
            'EDU_FATHER': [edu_father],
            'EDU_MOTHER': [edu_mother],
            'OCC_FATHER': [occ_father],
            'OCC_MOTHER': [occ_mother],
            'STRATUM': [stratum],
            'SISBEN': [sisben],
            'PEOPLE_HOUSE': [people_house],
            'INTERNET': [internet],
            'TV': [tv],
            'COMPUTER': [computer],
            'WASHING_MCH': [washing_machine],
            'MIC_OVEN': [microwave_oven],
            'CAR': [car],
            'DVD': [dvd],
            'FRESH': [fresh_food],
            'PHONE': [phone],
            'MOBILE': [mobile],
            'REVENUE': [family_revenue],
            'JOB': [job],
            'SCHOOL_NAT': [school_nat],
            'SCHOOL_TYPE': [school_type],
            'MAT_S11': [mat_s11],
            'CR_S11': [cr_s11],
            'CC_S11': [cc_s11],
            'BIO_S11': [bio_s11],
            'ENG_S11': [eng_s11],
            'ACADEMIC_PROGRAM': [academic_program],
            'QR_PRO': [qr_pro],
            'CR_PRO': [cr_pro],
            'CC_PRO': [cc_pro],
            'ENG_PRO': [eng_pro],
            'WC_PRO': [wc_pro],
            'FEP_PRO': [fep_pro],
            'G_SC': [g_sc],
            'PERCENTILE': [percentile],
            '2ND_DECILE': [second_decile], 
            'QUARTILE': [quartile],
            'SEL': [sel],
            'SEL_IHE': [sel_ihe],
            'Average_Score_Saber11': [average_score_saber11],
            'Performance_Saber11': [performance_saber11]  
        })

        # Preprocess input data
        input_data["EDU_FATHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
        input_data["EDU_MOTHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
        input_data["OCC_FATHER"].replace({"0": "Unemployed", "Home" : "Unemployed", "Retired": "Unemployed"}, inplace=True)
        input_data["OCC_MOTHER"].replace({"0": "Unemployed", "Home" : "Unemployed", "Retired": "Unemployed"}, inplace=True)
        input_data["SISBEN"].replace({"Esta clasificada en otro Level del SISBEN" : "It is classified in another SISBEN Level"}, inplace=True)
        input_data["PEOPLE_HOUSE"].replace({"Nueve" : "Nine", "Once" : "Eleven"}, inplace=True)
        input_data["JOB"].replace({"No":"0 hours per week","0":"0 hours per week"}, inplace=True)

        # Load trained model
        clf_rf = joblib.load('random_forest_model.joblib')

        # Preprocess input data
        input_data_encoded = encoder.transform(input_data[categorical_features])
        input_data_imputed = np.concatenate((numeric_imputer.transform(input_data[numerical_features]), input_data_encoded.toarray()), axis=1)
        input_data_scaled = scaler.transform(input_data_imputed)

        # Make prediction
        prediction = clf_rf.predict(input_data_scaled)

        # Extract lowest and highest numerical inputs
        input_numerical_values = {
            'Quantitative Reasoning': qr_pro,
            'Critical Reading': cr_pro,
            'Citizen Competencies': cc_pro,
            'English': eng_pro,
            'Written Communication': wc_pro,
            'Formulation of Engineering Projects': fep_pro,
            'Global Score': g_sc,
            'percentile': percentile,
            'Second Decile': second_decile,
            'Quartile': quartile,
            'Socioeconomic Level': sel,
            'Socioeconomic Level of The Institution of Higher Education': sel_ihe
        }
        # Extract lowest and highest numerical inputs with their corresponding values
        lowest_3 = nsmallest(3, input_numerical_values.items(), key=lambda x: x[1])
        highest_3 = nlargest(3, input_numerical_values.items(), key=lambda x: x[1])

        # Pass prediction result and input values to result template
        return render_template('result.html', prediction=prediction[0], lowest_3=lowest_3, highest_3=highest_3, input_data=input_data.to_dict(orient='records')[0], username=current_user.username)

if __name__ == '__main__':
    with app.app_context():
        inspector = db.inspect(db.engine)
        if 'user' in inspector.get_table_names():
            print("User table exists.")
        else: 
            print("User table does not exist.")
        
        db.create_all()
    
    app.run(debug=True)