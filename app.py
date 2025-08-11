import pickle
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session, send_file
import os
import sqlite3
import secrets
from werkzeug.utils import secure_filename 
import joblib
from fpdf import FPDF
from utils import svm_image_predict
from flask import Flask, render_template, request, jsonify
from chatbot.chatbot_response import get_bot_response,chatbot_knowledge

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Paths
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load behavioral model
behavior_model = joblib.load('models/behavior_model.pkl')

# Ensure folders exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

behavior_questions = [
    "Does the child respond to their name when called?",
    "Does the child make eye contact with others?",
    "Does the child enjoy social games?",
    "Does the child use gestures to communicate?",
    "Does the child engage in pretend play?",
    "Does the child point to objects of interest?",
    "Does the child understand simple instructions?",
    "Does the child share enjoyment with others?",
    "Does the child express a range of emotions?",
    "Does the child react appropriately to loud sounds?"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Username already exists")
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/select_mode')
def select_mode():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('select_mode.html', username=session.get('username'))



@app.route('/questions/<int:question_number>', methods=['GET', 'POST'])
def show_question(question_number):
    total_questions = len(behavior_questions)
    if question_number == 1:
        session['behavior_answers'] = []
    if request.method == 'POST':
        answer = int(request.form.get('answer'))
        session['behavior_answers'].append(answer)
        if question_number < total_questions:
            return redirect(url_for('show_question', question_number=question_number + 1))
        else:
            return redirect(url_for('behavior_result'))
    question_text = behavior_questions[question_number - 1]
    return render_template('behavior_questions.html',
                           question_number=question_number,
                           total_questions=total_questions,
                           question_text=question_text)

import traceback

@app.route('/behavior_result', methods=['POST'])
def behavior_result():
    # Collect form data
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    jaundice = request.form.get('jaundice')
    family_history = request.form.get('family_history')

    # Get the 10 behavioral question answers (Yes=1, No=0)
    questions = []
    for i in range(1, 11):
        ans = request.form.get(f'q{i}')
        questions.append(1 if ans.lower() == 'yes' else 0)

    # Prepare feature vector
    features = [age, 1 if gender == 'male' else 0,
                1 if jaundice == 'yes' else 0,
                1 if family_history == 'yes' else 0] + questions

    # Load model
    model = pickle.load(open('models/behavior_model.pkl', 'rb'))

    # Predict
    proba = model.predict_proba([features])[0]
    non_autistic_score = round(proba[0] * 100, 2)
    autistic_score = round(proba[1] * 100, 2)

    predicted_class_index = int(proba.argmax())
    predicted_label = model.classes_[predicted_class_index]
    confidence = round(proba[predicted_class_index] * 100, 2)

    # Chart data
    chart_labels = ["Autistic", "Non-Autistic"]
    chart_values = [autistic_score, non_autistic_score]

    return render_template(
        'behavior_result.html',
        confidence=confidence,
        chart_labels=chart_labels,
        chart_values=chart_values,
        prediction=predicted_label,
        autistic_score=autistic_score,
        non_autistic_score=non_autistic_score
    )


@app.route('/behavior')
def behavior():
    return render_template('behavior_questions.html')


@app.route('/image')
def image():
    return redirect(url_for('image_upload'))

@app.route('/image_upload')
def image_upload():
    return render_template('image_upload.html')

@app.route('/svm_image_predict', methods=['POST'])
def svm_image_predict_route():
    if 'image' not in request.files:
        return "No image part in request"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        try:
            label, confidence = svm_image_predict.predict_image(image_path)
            print(f"Prediction: {label}, Confidence: {confidence}")
            session['image_prediction'] = label
            session['image_score'] = round(confidence * 100, 2)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return f"Prediction failed: {str(e)}"
        relative_path = os.path.join('uploads', filename).replace("\\", "/")
        return render_template('image_result.html',
                               prediction=label,
                               confidence=round(confidence * 100, 2),
                               image_path=relative_path)
    return redirect(url_for('home'))



@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download_pdf')
def download_pdf():
    username = session.get('username', 'Guest')
    behavior = session.get('behavior_prediction', 'N/A')
    behavior_score = session.get('behavior_score', 0)
    image = session.get('image_prediction', 'N/A')
    image_score = session.get('image_score', 0)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Autism Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"User: {username}", ln=True)
    pdf.cell(200, 10, txt=f"Behavioral Result: {behavior} ({behavior_score}%)", ln=True)
    pdf.cell(200, 10, txt=f"Image Result: {image} ({image_score}%)", ln=True)
    report_path = "static/reports/report.pdf"
    pdf.output(report_path)
    return send_file(report_path, as_attachment=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
