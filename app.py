from flask import Flask, render_template, url_for, redirect, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user
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

########################################################################################################################################################################

# Load data and train Random Forest model
dataBase = pd.read_csv('Database/Academic Database.csv')
dataBase.drop(columns=['COD_S11', 'COD_SPRO', 'SCHOOL_NAME', 'UNIVERSITY', 'Unnamed: 9'], inplace=True)

####################################################################################
# Adding new columns:

# Calculate the average of SABER PRO'S 'QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO' metrics
dataBase["Average_Score_SABER_PRO"] = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']].mean(axis=1)
dataBase["Performance_SABER_PRO"] = np.select(
    [
        dataBase['Average_Score_SABER_PRO'] >= 90,
        dataBase['Average_Score_SABER_PRO'] >= 80,
        dataBase['Average_Score_SABER_PRO'] >= 70,
        dataBase['Average_Score_SABER_PRO'] >= 60
    ],
    [
        'Best',
        'Very Good',
        'Good',
        'Pass'
    ],
    default='Fail'
)

# Calculate the average of Saber 11 competences
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
    default='Fail'
)

####################################################################################
# Replace EDU_FATHER and EDU_MOTHER records of “Not sure” and “0” and “Ninguno” (meaning None) with “None”
dataBase["EDU_FATHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
dataBase["EDU_MOTHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)

# Replace OOC_FATHER and OCC_MOTHER records of “0”, “Home” and “Retires” with "unemployed"
dataBase["OCC_FATHER"].replace({"0": "Unemployed", "Home" : "Unemployed", "Retired": "Unemployed"}, inplace=True)
dataBase["OCC_MOTHER"].replace({"0": "Unemployed", "Home" : "Unemployed", "Retired": "Unemployed"}, inplace=True)

# Change “Esta clasificada en otro Level del SISBEN” into "It is classified in another SISBEN Level" from SISBEN
dataBase["SISBEN"].replace({"Esta clasificada en otro Level del SISBEN" : "It is classified in another SISBEN Level"}, inplace=True)

# In PEOPLE_HOUSE replace “Nueve” into “Nine” and “Once” into “Eleven”
dataBase["PEOPLE_HOUSE"].replace({"Nueve" : "Nine", "Once" : "Eleven"}, inplace=True)

# Replace “No” and “O” to “0 hours per week”
dataBase["JOB"].replace({"No":"0 hours per week","0":"0 hours per week"}, inplace=True)

####################################################################################

####################################################################################

# Data Splitting:
X_raw = dataBase[['GENDER', 'EDU_FATHER', 'EDU_MOTHER', 'OCC_FATHER', 'OCC_MOTHER',
       'STRATUM', 'SISBEN', 'PEOPLE_HOUSE', 'INTERNET', 'TV',
       'COMPUTER', 'WASHING_MCH', 'MIC_OVEN', 'CAR', 'DVD', 'FRESH', 'PHONE',
       'MOBILE', 'REVENUE', 'JOB', 'SCHOOL_NAT', 'SCHOOL_TYPE', 'MAT_S11',
       'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11', 'ACADEMIC_PROGRAM', 'QR_PRO',
       'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO', 'FEP_PRO', 'G_SC',
       'PERCENTILE', '2ND_DECILE', 'QUARTILE', 'SEL', 'SEL_IHE',
       'Average_Score_SABER_PRO','Average_Score_Saber11', 'Performance_Saber11']]

y_raw = dataBase["Performance_SABER_PRO"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

# Identify columns with missing values
missing_columns = X_train_raw.columns[X_train_raw.isnull().any()]

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number).columns
categorical_features = X_raw.select_dtypes(exclude=np.number).columns

# Handle missing values for numerical features
if not X_train_raw[numerical_features].empty:
    numeric_imputer = SimpleImputer(strategy="mean")
    X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features])
    X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features])
else:
    print("No numerical features to impute.")

# Handle missing values for categorical features
if not X_train_raw[categorical_features].empty:
    categoric_imputer = SimpleImputer(strategy="most_frequent")
    X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features])
    X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features])
else:
    print("No categorical features to impute.")

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')  # Ignores unknown categories
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed)
# For test data, use the categories learned from the training data
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed)

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded.toarray()), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded.toarray()), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train Random Forest classifier
clf_rf = RandomForestClassifier(bootstrap = False,
                                max_depth = 20,
                                min_samples_leaf = 1,
                                min_samples_split = 5,
                                n_estimators = 50)
clf_rf.fit(X_train_scaled, y_train)

# Define route to handle prediction requests
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Retrieve form data
        form_data = request.form
        # Get data from the form
        # Convert form data to appropriate data types and preprocess if needed
        gender = int(form_data.get('gender'))  # Assuming 'gender' is represented as an integer
        edu_father = form_data.get('edu_father')
        edu_mother = form_data.get('edu_mother')
        occ_father = form_data.get('occ_father')
        occ_mother = form_data.get('occ_mother')
        stratum = int(form_data.get('stratum'))  # Assuming 'stratum' is represented as an integer
        sisben = form_data.get('sisben')
        people_house = int(form_data.get('people_house'))  # Assuming 'people_house' is represented as an integer
        internet = int(form_data.get('internet'))  # Assuming 'internet' is represented as an integer (e.g., 0 or 1)
        tv = int(form_data.get('tv'))  # Assuming 'tv' is represented as an integer (e.g., 0 or 1)
        computer = int(form_data.get('computer'))  # Assuming 'computer' is represented as an integer (e.g., 0 or 1)
        washing_machine = int(form_data.get('washing_machine'))  # Assuming 'washing_machine' is represented as an integer (e.g., 0 or 1)
        microwave_oven = int(form_data.get('microwave_oven'))  # Assuming 'microwave_oven' is represented as an integer (e.g., 0 or 1)
        car = int(form_data.get('car'))  # Assuming 'car' is represented as an integer (e.g., 0 or 1)
        dvd = int(form_data.get('dvd'))  # Assuming 'dvd' is represented as an integer (e.g., 0 or 1)
        phone = int(form_data.get('phone'))  # Assuming 'phone' is represented as an integer (e.g., 0 or 1)
        mobile = int(form_data.get('mobile'))  # Assuming 'mobile' is represented as an integer (e.g., 0 or 1)
        fresh_food = int(form_data.get('fresh_food'))  # Assuming 'fresh_food' is represented as an integer (e.g., 0 or 1)
        family_revenue = float(form_data.get('family_revenue'))  # Assuming 'family_revenue' is represented as a floating-point number
        job = form_data.get('job')
        school_nat = form_data.get('school_nat')
        school_type = form_data.get('school_type')
        mat_s11 = float(form_data.get('mat_s11'))  # Assuming 'mat_s11' is represented as a floating-point number
        cr_s11 = float(form_data.get('cr_s11'))  # Assuming 'cr_s11' is represented as a floating-point number
        cc_s11 = float(form_data.get('cc_s11'))  # Assuming 'cc_s11' is represented as a floating-point number
        bio_s11 = float(form_data.get('bio_s11'))  # Assuming 'bio_s11' is represented as a floating-point number
        eng_s11 = float(form_data.get('eng_s11'))  # Assuming 'eng_s11' is represented as a floating-point number
        average_score_saber11 = float(form_data.get('average_score_saber11'))  # Assuming 'average_score_saber11' is represented as a floating-point number
        performance_saber11 = form_data.get('performance_saber11')

        # Make predictions using the trained model
        input_data = np.array([[gender, edu_father, edu_mother, occ_father, occ_mother, stratum, sisben, people_house, 
                                internet, tv, computer, washing_machine, microwave_oven, car, dvd, phone, mobile, 
                                fresh_food, family_revenue, job, school_nat, school_type, mat_s11, cr_s11, cc_s11, 
                                bio_s11, eng_s11, average_score_saber11, performance_saber11]])

        # Preprocess the input data
        # Handle missing values for numerical features
        input_data_numerical = numeric_imputer.transform(input_data[:, [22, 23, 24, 25, 26, 27]])
        # Handle missing values for categorical features
        input_data_categorical = categoric_imputer.transform(input_data[:, [1, 2, 3, 4, 6]])
        # Encode categorical features
        input_data_encoded = encoder.transform(input_data_categorical)
        # Concatenate numerical and encoded categorical features
        input_data_scaled = scaler.transform(np.concatenate((input_data_numerical, input_data_encoded.toarray()), axis=1))

        # Make prediction using the trained model
        prediction = clf_rf.predict(input_data_scaled)

        # Do something with the prediction
        # For example, render a template with the prediction
        return render_template('result.html', prediction_results={'prediction': prediction})

def map_prediction_to_category(prediction):
    if prediction == 'Best':
        return 'Top performer'
    elif prediction == 'Very Good':
        return 'High achiever'
    elif prediction == 'Good':
        return 'Above average'
    elif prediction == 'Pass':
        return 'Average'
    elif prediction == 'Fail':
        return 'Below average'
    else:
        return 'Critical intervention needed'

@app.route('/result', methods=['GET', 'POST'])
@login_required
def result():
    if request.method == 'POST':
        # Handle any form submissions if necessary
        pass  # Placeholder, you can add handling logic here
    
    # Assuming prediction_results is defined elsewhere in your code
    prediction_results = {'prediction': 'some_value'}  # Replace 'some_value' with the actual prediction
    prediction_category = map_prediction_to_category(prediction_results['prediction'])
    
    return render_template('result.html', prediction_category=prediction_category)


if __name__ == '__main__':
    with app.app_context():
        inspector = db.inspect(db.engine)
        if 'user' in inspector.get_table_names():
            print("User table exists.")
        else:
            print("User table does not exist.")
        
        db.create_all()
    
    app.run(debug=True)