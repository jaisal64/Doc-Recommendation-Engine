import os
import sys
# Add the parent directory to sys.path to find the models package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
# Now import your model class from the models package
from models.doctor_recommendation_model import DoctorRecommendationModel
from config import DB_CONFIG

app = Flask(__name__)

# Load the model - assuming your working directory is backend_ml when you run the Flask app
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'doctor_recommendation_model.pkl')
# model = DoctorRecommendationModel.load_model(model_path)

@app.route('/recommend-doctors', methods=['POST'])
def recommend_doctors():

    data = request.get_json(force=True)
    patient_id = data['patient_id']
    top_n = data.get('top_n', 5)  # Default to 5 recommendations if not specified

    # Use the model to get recommendations
    recommended_doctors = model.recommend_doctors(patient_id, top_n=top_n)

    # Convert to desired format and return
    return jsonify([doctor.to_dict() for doctor in recommended_doctors])
    
    # data = request.get_json(force=True)

    # patient_index = data['patient_index']
    
    # # Initialize your model (consider modifying the model to accept DB config or connection)
    # model = DoctorRecommendationModel()
    
    # # Use the model to get recommendations
    # recommended_doctors = model.recommend_doctors(patient_index, top_n=5)
    
    # # Convert to desired format and return
    # return jsonify(recommended_doctors)




    # patient_index = data['patient_index']
    # top_n = data.get('top_n', 5)  # Default to 5 recommendations if not specified
    
    # # Get recommendations
    # recommended_doctors = model.recommend_doctors(patient_index, top_n=top_n)
    
    # # Convert recommendations to a list of dicts (or similar, depending on your data structure)
    # recommendations = recommended_doctors.to_dict('records')
    
    # return jsonify(recommendations)


@app.route('/get-patients', methods=['GET'])
def get_patients():
    try:
        patients_data = model.fetch_patient_data()
        return jsonify(patients_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
