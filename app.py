from flask import Flask, render_template, request
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


app = Flask(__name__)

model_file = 'model.pkl'
absolute_model_path = os.path.abspath(model_file)
rf_model = joblib.load(open(absolute_model_path, 'rb'))


def predict_oral_cancer_cl(data):
    try:
        if data.shape[1] != len(rf_model.feature_importances_):
            raise ValueError("Input data shape does not match model's feature count.")
        
        predictions = rf_model.predict(data)
        return round(predictions[0], 3)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model_name.h5')
class_names = ['cancer', 'non-cancer']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_flask(img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess input based on ResNet50 requirements
    return img_array

def create_upload_directory():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def predict_oral_cancer_img(img_path):
    
    print("DEBUG: Preprocessing image...")
    img_array = preprocess_image_flask(img_path)
    print("DEBUG: Image preprocessing completed.")
    
    print("DEBUG: Performing prediction...")
    pred = model.predict(img_array)
    print("DEBUG: Prediction completed.")
    
    predicted_class_index = np.argmax(pred)
    predicted_class = class_names[predicted_class_index]
    
    print("DEBUG: Predicted Class:", predicted_class)
    print("DEBUG: Prediction Probabilities:", pred.flatten())
    
    return predicted_class, pred.flatten()

def get_hospitals_by_district(district, max_hospitals=5):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv('hospi.csv')

        # Filter hospitals by district
        filtered_hospitals = df[df['District'] == district]

        # Get up to max_hospitals from the filtered DataFrame
        hospitals_list = []
        for index, row in filtered_hospitals.iterrows():
            hospital_details = {
                'Name': row['Name'],
                'Contact': row['Contact'],
                'Address': row['Address']
                # Add more details if needed
            }
            hospitals_list.append(hospital_details)

            # Break the loop if reached maximum hospitals
            if len(hospitals_list) >= max_hospitals:
                break

        return hospitals_list
    except Exception as e:
        print(f"Error loading hospitals: {e}")
        return []

   
@app.route('/', methods=['GET', 'POST'])
def home():
    create_upload_directory()
    if request.method == 'POST':
        input_data = {
            'localization': int(request.form['localization']),
            'size': request.form['size'],
            'tobacco_use': int(request.form['tobacco_use']),
            'alcohol_consumption': int(request.form['alcohol_consumption']),
            'sun_exposure': int(request.form['sun_exposure']),
            'gender': int(request.form['gender']),
            'age_group': int(request.form['age_group'])
        }

        input_df = pd.DataFrame([input_data])

        district = request.form['district']
        hospitals_list = get_hospitals_by_district(district)
        result_cl = predict_oral_cancer_cl(input_df)

        if 'file' not in request.files:
            print("DEBUG: No file part in POST request")
            return render_template('index.php', error="No file part")

        file = request.files['file']
        print("DEBUG: Filename in GET request:", file.filename)
        if file.filename == '':
            print("DEBUG: No selected file in POST request")
            return render_template('index.php', error="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("DEBUG: File saved:", filename)

            result_img = predict_oral_cancer_img(file_path)
            predicted_class, probabilities = predict_oral_cancer_img(file_path)
            print("DEBUG: Predicting oral cancer for:", filename)

            os.remove(file_path)
            print("DEBUG: Removed uploaded file:", filename)
            
            return render_template('result.php', result_cl=result_cl,class_names=class_names,predicted_class=predicted_class, hospitals_list=hospitals_list)

    return render_template('index.php')
@app.route('/causes')
def causes():
    return render_template('causes.html')
@app.route('/about_oc')
def about_oc():
    return render_template('about_oc.html')
@app.route('/treatment')
def treatment():
    return render_template('treatment.html')
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

   


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)  # Specify the port (e.g., 8000) and host
