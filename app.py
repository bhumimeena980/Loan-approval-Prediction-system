from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    applicant_income = float(request.form['income'])
    loan_amount = float(request.form['loanamt'])
    credit_history = int(request.form['credithistory'])

    features = np.array([[gender, married, applicant_income, loan_amount, credit_history]])
    prediction = model.predict(features)

    output = "Approved" if prediction[0] == 1 else "Rejected"
    return render_template('index.html', result=f"Loan Status: {output}")

if __name__ == '__main__':
    app.run(debug=True)
