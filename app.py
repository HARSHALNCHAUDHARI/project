import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a strong secret key

# In-memory user database for demonstration (replace with a real database for production)
users = {}

# Load pre-trained Keras model
try:
    model = load_model('/Users/seflame/Desktop/Project/keras_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username]['password'] == password:
            session['logged_in'] = True
            return redirect(url_for('learning'))  # Redirect to learning page after login
        else:
            return render_template('login.html', error='Invalid credentials.')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        mobile = request.form.get('mobile')

        if username in users:
            return render_template('signup.html', error='Username already exists.')

        # Save the new user
        users[username] = {'password': password, 'mobile': mobile}
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

@app.route('/prediction')
def prediction():
    if 'logged_in' in session:
        return render_template('prediction.html')
    return redirect(url_for('login'))

@app.route('/learning')
def learning():
    # Ensure the user is logged in to access the learning page
    if 'logged_in' in session:
        return render_template('learning.html')
    return redirect(url_for('login'))

@app.route('/fetch-data', methods=['POST'])
def fetch_data():
    data = request.json
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return jsonify({'error': 'No data found for the given ticker and date range.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    stats = df.describe().round(2).to_dict()

    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []
    for i in range(100, data_training_array.shape[0]): 
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    try:
        y_predicted = model.predict(x_test)
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted.flatten() * scale_factor
    y_test = y_test * scale_factor

    data_json = df.reset_index().to_dict(orient='records')

    return jsonify({
        'data': data_json,
        'predictions': y_predicted.tolist(),
        'actual': y_test.tolist(),
        'statistics': stats
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
