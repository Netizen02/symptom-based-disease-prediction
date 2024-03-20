from flask import Flask, request,jsonify
import joblib
import numpy as np
from statistics import mode
from flask_cors import CORS,  cross_origin


app = Flask(__name__)
CORS(app)

# More Headers if CORS Error Persist
# app.config["CORS_HEADERS"] = "Content-Type"
# CORS(app, resources={r"/*": {"origins": "*"}})

# @app.after_request
# def add_cors_headers(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     return response

# load the pickle files using joblib.load
final_rf_model = joblib.load('model_rf.pkl')
final_nb_model = joblib.load('model_nb.pkl')
final_svm_model = joblib.load('model_svm.pkl')
encoder = joblib.load('encoder.pkl')

def initializing():
    symptoms=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
    'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
    'swelling_joints', 'movement_stiffness', 'spinning_movements',
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
    'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking',
    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    
    print("Just Checking")
    print(symptoms)

    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
    }
    
    return data_dict

@app.route('/process', methods=['POST','GET'])
@cross_origin(origin='*', headers=['Content-Type','Authorization'])
def process_data():

    symptoms = request.json['input']  
    print("Request data")
    print(symptoms)
    
    data_dict = initializing()
    
    # Comma Splitting
    symptoms = symptoms.split(",")
    print("Data after splitting by , ")
    print(symptoms)

    input_data = [0] * len(data_dict["symptom_index"])
    print("Input data should be expanded to 135 features")
    print(input_data)

    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    print("Input data before reshape")
    print(input_data)
	# reshaping the input data and converting it
	# into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    print("Input data after reshape")
    print(input_data)
    
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)

