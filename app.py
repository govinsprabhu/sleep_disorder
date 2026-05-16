from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model and Encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)
        # We need these features: 'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'BloodPressure_Upper_Value', 'BloodPressure_Lower_Value'
        
        # Parse inputs
        gender = str(data.get('Gender'))
        age = float(data.get('Age'))
        occupation = str(data.get('Occupation'))
        sleep_duration = float(data.get('Sleep Duration'))
        quality_of_sleep = float(data.get('Quality of Sleep'))
        phys_activity = float(data.get('Physical Activity Level'))
        stress_level = float(data.get('Stress Level'))
        bmi_category = str(data.get('BMI Category'))
        heart_rate = float(data.get('Heart Rate'))
        daily_steps = float(data.get('Daily Steps'))
        
        bp = str(data.get('Blood Pressure'))
        if '/' in bp:
            bp_parts = bp.split('/')
            bp_upper = float(bp_parts[0])
            bp_lower = float(bp_parts[1])
        else:
            bp_upper = 120.0
            bp_lower = 80.0
            
        def encode_feature(col_name, val):
            le = label_encoders[col_name]
            if val not in le.classes_:
                val = le.classes_[0] 
            return int(le.transform([val])[0])

        gender_encoded = encode_feature('Gender', gender)
        occupation_encoded = encode_feature('Occupation', occupation)
        bmi_encoded = encode_feature('BMI Category', bmi_category)

        input_data = pd.DataFrame([{
            'Gender': gender_encoded,
            'Age': age,
            'Occupation': occupation_encoded,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': phys_activity,
            'Stress Level': stress_level,
            'BMI Category': bmi_encoded,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'BloodPressure_Upper_Value': bp_upper,
            'BloodPressure_Lower_Value': bp_lower,
        }])
            
        # Predict
        print(input_data['Occupation'])
        prediction_encoded = model.predict(input_data)[0]
        
        # Decode
        le_target = label_encoders['Sleep Disorder']
        prediction_label = le_target.inverse_transform([prediction_encoded])[0]
        
        return jsonify({'prediction': str(prediction_label), 'status': 'success'})
        
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
