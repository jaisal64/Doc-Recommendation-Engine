import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class DoctorRecommendationModel:
    def __init__(self):
        self.patients_df = None
        self.doctors_df = None
        self.patient_features = None
        self.scaler = MinMaxScaler()
        self.ohe_sex_location = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    def fit(self, patients_df, doctors_df):
        # Preprocess and align features
        self.patients_df, self.doctors_df = self.preprocess_align_features(patients_df, doctors_df)
        # Prepare feature vectors for patients
        self.prepare_patient_features()
        
    def preprocess_align_features(self, patients_df, doctors_df):
        # Combine, encode, and scale features (as previously described)
        # This method should include all the preprocessing steps implemented before

        self.scaler = MinMaxScaler()

        doctors_df['language_combined'] = doctors_df['language_1'].fillna('') + ',' + doctors_df['language_2'].fillna('')
    
        # Initialize OneHotEncoder
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # Encode 'sex' and 'location' using LabelEncoder for simplicity
        for column in ['sex', 'location']:
            le = LabelEncoder()
            combined = pd.concat([patients_df[column].fillna('Unknown'), doctors_df[column].fillna('Unknown')])
            le.fit(combined)
            patients_df[column + '_encoded'] = le.transform(patients_df[column].fillna('Unknown'))
            doctors_df[column + '_encoded'] = le.transform(doctors_df[column].fillna('Unknown'))
        
        # Convert 'budget_max' and 'cost_max' to numeric, ensuring columns exist and are not null before attempting string operations
        if 'budget_max' in patients_df.columns and patients_df['budget_max'].dtype == object:
            patients_df['budget_max'] = pd.to_numeric(patients_df['budget_max'].str.replace('[\$,]', '', regex=True), errors='coerce').fillna(0)
        
        if 'cost_max' in doctors_df.columns and doctors_df['cost_max'].dtype == object:
            doctors_df['cost_max'] = pd.to_numeric(doctors_df['cost_max'].str.replace('[\$,]', '', regex=True), errors='coerce').fillna(0)
        
        combined_budget_cost = np.concatenate((patients_df[['budget_max']].values, doctors_df[['cost_max']].values), axis=0)
        self.scaler.fit(combined_budget_cost)

        # Handle insurance by creating a simplified matching column for each patient's insurance in the doctors DataFrame
        unique_insurances = patients_df['Insurance_plan'].dropna().unique()
        for insurance in unique_insurances:
            insurance_column = 'accepts_' + insurance.replace(' ', '_').replace('/', '_')
            doctors_df[insurance_column] = doctors_df.apply(lambda x: 1 if insurance in x.values else 0, axis=1)

        patients_df['budget_max_scaled'] = self.scaler.transform(patients_df[['budget_max']])
        doctors_df['cost_max_scaled'] = self.scaler.transform(doctors_df[['cost_max']])
    
        
        return patients_df, doctors_df

    def prepare_patient_features(self):
        # Prepare the patient feature vectors for similarity computation
        
        self.patient_features = np.hstack([
        self.patients_df[['sex_encoded', 'location_encoded']].to_numpy(),
        self.patients_df[['budget_max_scaled']].to_numpy()  # Use the scaled version
    ])
        
    def recommend_doctors(self, patient_id, top_n=5):
        # Use self to refer to instance variables
        avg_budget = self.patients_df.loc[patient_id, 'budget_max_scaled']
        
        # Filter doctors based on the average budget (scaled) of similar patients
        suitable_doctors = self.doctors_df[self.doctors_df['cost_max_scaled'] <= avg_budget]
        
        # Compute similarity between the patient preferences and doctors using self.patient_features
        doctor_similarity = cosine_similarity([self.patient_features[patient_id]], suitable_doctors[['sex_encoded', 'location_encoded', 'cost_max_scaled']].to_numpy())
        
        # Find top N indices of doctors based on similarity scores
        top_doctor_indices = np.argsort(-doctor_similarity[0])[:top_n]
        recommended_doctors = suitable_doctors.iloc[top_doctor_indices]
        
        return recommended_doctors[['first_name', 'last_name', 'speciality', 'location', 'cost_max']]


    def save_model(self, file_path):
        # Save the model components to a file
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        # Load the model components from a file
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model

model = DoctorRecommendationModel()

patients_df = pd.read_csv('../data/patients_data (1).csv')
doctors_df = pd.read_csv('../data/doctors-data.csv')

model.fit(patients_df, doctors_df)  
model.save_model('doctor_recommendation_model.pkl')

loaded_model = DoctorRecommendationModel.load_model('doctor_recommendation_model.pkl')

# Replace 0 with any valid patient index from your dataset
patient_index = 5
top_n = 4  # Number of doctor recommendations you want

# Get recommendations
recommended_doctors = loaded_model.recommend_doctors(patient_index, top_n=top_n)

# Print the recommendations
print(recommended_doctors)