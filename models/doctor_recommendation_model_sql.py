# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# import numpy as np
# import mysql.connector

# class DoctorRecommendationModel:
#     def __init__(self, db_config):
#         self.db_connection = mysql.connector.connect(
#             host=db_config['host'],
#             user=db_config['user'],
#             password=db_config['password'],
#             database=db_config['database']
#         )
#         self.model = self.load_model('./doctor_recommendation_model.pkl')
#         self.label_encoders = {}
#         self.scaler = StandardScaler()
    
#     def load_model(self, model_path):
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         return model

#     def fetch_data(self):
#         try:
#             cursor = self.db_connection.cursor(dictionary=True)
#             cursor.execute("SELECT * FROM patients")
#             patients_data = cursor.fetchall()
#             print(f"Fetched {len(patients_data)} patients")

#             cursor.execute("SELECT * FROM doctors")
#             doctors_data = cursor.fetchall()
#             print(f"Fetched {len(doctors_data)} doctors")

#             if len(patients_data) == 0 or len(doctors_data) == 0:
#                 print("Warning: One or more tables are empty.")

#             return pd.DataFrame(patients_data), pd.DataFrame(doctors_data)
#         except mysql.connector.Error as err:
#             print(f"Database error: {err}")
#             return pd.DataFrame(), pd.DataFrame()

#     def preprocess_data(self, df):
#         # Handle missing values
#         df.fillna(df.mean(), inplace=True)
        
#         # Encoding categorical features
#         for column in ['sex', 'location']:
#             if column not in self.label_encoders:
#                 self.label_encoders[column] = LabelEncoder()
#                 df[column] = self.label_encoders[column].fit_transform(df[column])
#             else:
#                 df[column] = self.label_encoders[column].transform(df[column])

#         # Scaling numerical features
#         numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
#         df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
#         return df

#     def get_recommendations(self, patient_features):
#         patients_df, doctors_df = self.fetch_data()
        
#         # Preprocess the data
#         patients_df = self.preprocess_data(patients_df)
#         doctors_df = self.preprocess_data(doctors_df)

#         # Assuming patient_features is a dict with patient information
#         patient_df = pd.DataFrame([patient_features])
#         patient_df = self.preprocess_data(patient_df)

#         # Use the model to predict scores or similarities
#         patient_features_array = patient_df.iloc[0].values.reshape(1, -1)
#         doctors_features = doctors_df.values
        
#         # Here, you might use your model to predict; this is a placeholder for model usage
#         # For demonstration, using cosine similarity as a proxy for model predictions
#         similarities = cosine_similarity(patient_features_array, doctors_features)
#         top_doctor_indices = np.argsort(similarities[0])[::-1][:5]  # Top 5 recommendations
        
#         # Fetch top doctor details
#         recommended_doctors = doctors_df.iloc[top_doctor_indices]
#         return recommended_doctors.to_dict('records')

# def test_recommendations():
#         db_config = {
#             'user': 'root',
#             'password': '&GyUJ#bpc+ZzgYYd',
#             'host': '35.222.184.202',
#             'database': 'orig',
#             # 'raise_on_warnings': True
#         }
#         model = DoctorRecommendationModel(db_config)
#         patient_features = {
#             'age': 30,
#             'sex': 'Male',  # Assuming your LabelEncoder is trained with 'Male' as one of the classes
#             'location': 'Brighton'  # And 'New York' is a valid location in your dataset
#         }
#         recommendations = model.get_recommendations(patient_features)
#         print("Top 4 recommended doctors for the patient:")
#         for i, doctor in enumerate(recommendations[:4], start=1):
#             print(f"{i}: Doctor ID {doctor['id']}, Name: {doctor['name']}, Specialty: {doctor['specialty']}, Location: {doctor['location']}")


# # Example usage
# if __name__ == "__main__":
# #     db_config =  {
# #         'user': 'root',
# #         'password': '&GyUJ#bpc+ZzgYYd',
# #         'host': '35.222.184.202',
# #         'database': 'orig',
# #         'raise_on_warnings': True
# # }
# #     model = DoctorRecommendationModel(db_config)
#     test_recommendations()

