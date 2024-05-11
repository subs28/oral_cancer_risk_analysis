from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import joblib
 # Use joblib for compatibility with older versions of scikit-learn

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the machine learning models and other required files
model_cl = joblib.load('model.pkl')
model_img = load_model('model_name.h5')
class_names = ['cancer', 'non-cancer']

def preprocess_image(img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess input based on ResNet50 requirements
    return img_array

def predict_oral_cancer_cl(data):
    try:
        predictions = model_cl.predict(data)
        return round(predictions[0], 3)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def predict_oral_cancer_img(img_path):
    try:
        img_array = preprocess_image(img_path)
        pred = model_img.predict(img_array)
        predicted_class_index = np.argmax(pred)
        predicted_class = class_names[predicted_class_index]
        return predicted_class, pred.flatten()
    except Exception as e:
        print(f"Error during image prediction: {e}")
        return None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def get_hospitals_by_district(district, max_hospitals=5):
    try:
        df = pd.read_csv('hospi.csv')
        filtered_hospitals = df[df['District'] == district]
        hospitals_list = []
        for index, row in filtered_hospitals.iterrows():
            hospital_details = {
                'Name': row['Name'],
                'Contact': row['Contact'],
                'Address': row['Address']
            }
            hospitals_list.append(hospital_details)
            if len(hospitals_list) >= max_hospitals:
                break
        return hospitals_list
    except Exception as e:
        print(f"Error loading hospitals: {e}")
        return []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle form submission
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
        result_cl = predict_oral_cancer_cl(input_df)
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class, probabilities = predict_oral_cancer_img(file_path)
            os.remove(file_path)
            district = request.form['district']
            hospitals_list = get_hospitals_by_district(district)
            
            return render_template('result.php', result_cl=result_cl, class_names=class_names, 
                                   predicted_class=predicted_class, hospitals_list=hospitals_list)
    
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
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
