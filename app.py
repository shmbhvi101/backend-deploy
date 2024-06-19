import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import sqlite3

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'continuous_feel_of_urine',
      'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 
      'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 
      'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 
      'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 
      'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
           'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes',
           'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine',
           'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
           'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
           'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
           'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
           'Osteoarthristis', 'Arthritis',
           '(vertigo) Paroymsal Positional Vertigo', 'Acne',
           'Urinary tract infection', 'Psoriasis', 'Impetigo']

file_path = "C:/Users/Shambhavi/Downloads/int247-machine-learning-project-2020-kem031-sudhanshu (extract.me)/proj2/Dataset/training.csv"
df = pd.read_csv(file_path)

# Encode the target labels into numerical values
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

# Split data into features and target
X = df[l1]
y = df['prognosis']

# Train models
dt_model = tree.DecisionTreeClassifier().fit(X, y)
rf_model = RandomForestClassifier(n_estimators=100).fit(X, y)
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2).fit(X, y)
nb_model = GaussianNB().fit(X, y)

models = {
    'decision_tree': dt_model,
    'random_forest': rf_model,
    'knn': knn_model,
    'naive_bayes': nb_model
}

def store_prediction(name, symptoms, predicted_disease, model_name):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {model_name} (
        Name TEXT,
        Symptom1 TEXT,
        Symptom2 TEXT,
        Symptom3 TEXT,
        Symptom4 TEXT,
        Symptom5 TEXT,
        Disease TEXT
    )
    ''')
    
    cursor.execute(f'''
    INSERT INTO {model_name} (Name, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, Disease)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, symptoms[0], symptoms[1], symptoms[2], symptoms[3], symptoms[4], predicted_disease))
    
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    name = data.get('name', '')
    symptoms = [data.get('symptom1', ''), data.get('symptom2', ''), data.get('symptom3', ''), data.get('symptom4', ''), data.get('symptom5', '')]

    if not name or all(symptom == "" for symptom in symptoms):
        return jsonify({"error": "Please enter a name and at least one symptom."}), 400

    input_features = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            input_features[l1.index(symptom)] = 1

    input_features = [input_features]
    model_name = data.get('model', 'decision_tree')
    if model_name not in models:
        return jsonify({"error": "Invalid model name."}), 400

    model = models[model_name]
    predicted_disease_index = model.predict(input_features)[0]
    predicted_disease = disease[predicted_disease_index]

    # Store the prediction in the database
    store_prediction(name, symptoms, predicted_disease, model_name)
    
    return jsonify({"predicted_disease": predicted_disease})

if __name__ == '__main__':
    app.run(debug=True)
