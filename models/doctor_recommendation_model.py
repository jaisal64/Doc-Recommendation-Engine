import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import mysql.connector

class DoctorRecommendationModel:
    def __init__(self, db_config):
        self.db_config = db_config
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.connect_to_db()

    def connect_to_db(self):
        self.connection = mysql.connector.connect(**self.db_config)
        self.cursor = self.connection.cursor(dictionary=True)

    def fetch_data(self, query):
        self.cursor.execute(query)
        return pd.DataFrame(self.cursor.fetchall())

    def preprocess_data(self, df, fit_scaler=False):
        # Encode categorical features
        for column in ['sex', 'location']:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column + '_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                df[column + '_encoded'] = self.label_encoders[column].transform(df[column])

        # Scale numerical features
        numerical_features = df[['budget_max' if 'budget_max' in df.columns else 'cost_max']].values
        if fit_scaler:
            self.scaler.fit(numerical_features)
        df['budget_cost_scaled'] = self.scaler.transform(numerical_features)

        return df

    def recommend_doctors(self, patient_id, top_n=5):
        # Fetch and preprocess patient data
        patient_query = f"SELECT * FROM Patients WHERE patient_id = {patient_id}"
        patient_df = self.preprocess_data(self.fetch_data(patient_query), fit_scaler=True)

        # Fetch and preprocess doctor data
        doctors_df = self.preprocess_data(self.fetch_data("SELECT * FROM Doctors"))

        # Compute similarity between patient preferences and all doctors
        patient_features = patient_df[['sex_encoded', 'location_encoded', 'budget_cost_scaled']].to_numpy()
        doctor_features = doctors_df[['sex_encoded', 'location_encoded', 'cost_max_scaled']].to_numpy()
        similarity_scores = cosine_similarity(patient_features, doctor_features)

        # Find top N doctors based on similarity scores
        top_indices = np.argsort(-similarity_scores[0])[:top_n]
        recommended_doctors = doctors_df.iloc[top_indices]

        return recommended_doctors[['doctor_id', 'first_name', 'last_name', 'speciality', 'location', 'cost_max']]

    def close_connection(self):
        self.cursor.close()
        self.connection.close()

# Database configuration
db_config = {
        'user': 'root',
        'password': '&GyUJ#bpc+ZzgYYd',
        'host': '35.222.184.202',
        'database': 'orig',
        'raise_on_warnings': True
}


# Example usage
model = DoctorRecommendationModel(db_config = db_config)
recommendations = model.recommend_doctors(patient_id=1, top_n=5)
print(recommendations)
model.close_connection()


# Commented out the pickling operations
# The model instantiation and usage should now directly interact with the MySQL database.























# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import pickle

# import mysql.connector
# from mysql.connector import Error

# class DoctorRecommendationModel:
#     def __init__(self):
        
#         self.patients_df = None
#         self.doctors_df = None
#         self.patient_features = None
#         self.db_config = {
#         'user': 'root',
#         'password': '&GyUJ#bpc+ZzgYYd',
#         'host': '35.222.184.202',
#         'database': 'orig',
#         'raise_on_warnings': True
#     }
#         self.scaler = MinMaxScaler()
#         self.ohe_sex_location = OneHotEncoder(sparse=False, handle_unknown='ignore')
#         # self.db_config = db_config  # Store database configuration
#         self.cnx = mysql.connector.connect(**self.db_config)
#         self.cursor = self.cnx.cursor(dictionary=True)
        
#         self.connect_to_db()

#     def __getstate__(self):
#         state = self.__dict__.copy()
#         # Remove the unpickleable entries.
#         del state['cnx']
#         del state['cursor']
#         return state

#     def __setstate__(self, state):
#         # Restore instance attributes.
#         self.__dict__.update(state)
#         # Reinitialize the database connection after unpickling.
#         self.connect_to_db()
        
        
        
#     def fit(self, patients_df, doctors_df):
#         # Preprocess and align features
#         self.patients_df, self.doctors_df = self.preprocess_align_features(patients_df, doctors_df)
#         # Prepare feature vectors for patients
#         self.prepare_patient_features()

#     def connect_to_db(self):
#         try:
#             self.cnx = mysql.connector.connect(**self.db_config)
#             if self.cnx.is_connected():
#                 self.cursor = self.cnx.cursor(dictionary=True)  # Use dictionary cursor to easily manipulate data
#                 print("Successfully connected to the database")
#         except Error as e:
#             print(f"Error connecting to MySQL: {e}")

#     def fetch_patient_data(self):
#         self.cursor.execute("SELECT * FROM Patients")
#         return self.cursor.fetchall()

#     def fetch_doctor_data(self):
#         # Example method to fetch doctor data
#         self.cursor.execute("SELECT * FROM Doctors")
#         return self.cursor.fetchall()

#     def close_connection(self):
#         if self.cnx.is_connected():
#             self.cursor.close()
#             self.cnx.close()
#             print("MySQL connection is closed")
        
#     def preprocess_align_features(self, patients_df, doctors_df):
#         # Combine, encode, and scale features (as previously described)
#         # This method should include all the preprocessing steps implemented before

#         self.scaler = MinMaxScaler()

#         doctors_df['language_combined'] = doctors_df['language_1'].fillna('') + ',' + doctors_df['language_2'].fillna('')
    
#         # Initialize OneHotEncoder
#         ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
#         # Encode 'sex' and 'location' using LabelEncoder for simplicity
#         for column in ['sex', 'location']:
#             le = LabelEncoder()
#             combined = pd.concat([patients_df[column].fillna('Unknown'), doctors_df[column].fillna('Unknown')])
#             le.fit(combined)
#             patients_df[column + '_encoded'] = le.transform(patients_df[column].fillna('Unknown'))
#             doctors_df[column + '_encoded'] = le.transform(doctors_df[column].fillna('Unknown'))
        
#         # Convert 'budget_max' and 'cost_max' to numeric, ensuring columns exist and are not null before attempting string operations
#         if 'budget_max' in patients_df.columns and patients_df['budget_max'].dtype == object:
#             patients_df['budget_max'] = pd.to_numeric(patients_df['budget_max'].str.replace('[\$,]', '', regex=True), errors='coerce').fillna(0)
        
#         if 'cost_max' in doctors_df.columns and doctors_df['cost_max'].dtype == object:
#             doctors_df['cost_max'] = pd.to_numeric(doctors_df['cost_max'].str.replace('[\$,]', '', regex=True), errors='coerce').fillna(0)
        
#         combined_budget_cost = np.concatenate((patients_df[['budget_max']].values, doctors_df[['cost_max']].values), axis=0)
#         self.scaler.fit(combined_budget_cost)

#         # Handle insurance by creating a simplified matching column for each patient's insurance in the doctors DataFrame
#         unique_insurances = patients_df['Insurance_plan'].dropna().unique()
#         for insurance in unique_insurances:
#             insurance_column = 'accepts_' + insurance.replace(' ', '_').replace('/', '_')
#             doctors_df[insurance_column] = doctors_df.apply(lambda x: 1 if insurance in x.values else 0, axis=1)

#         patients_df['budget_max_scaled'] = self.scaler.transform(patients_df[['budget_max']])
#         doctors_df['cost_max_scaled'] = self.scaler.transform(doctors_df[['cost_max']])
    
        
#         return patients_df, doctors_df

#     def prepare_patient_features(self):
#         # Prepare the patient feature vectors for similarity computation
        
#         self.patient_features = np.hstack([
#         self.patients_df[['sex_encoded', 'location_encoded']].to_numpy(),
#         self.patients_df[['budget_max_scaled']].to_numpy()  # Use the scaled version
#     ])
        
#     def recommend_doctors(self, patient_index, top_n=5):
#         # Use self to refer to instance variables

#         # Fetch patient data by patient_id
#         patient_data = self.fetch_patient_data_by_id(patient_id)
#         # Convert patient data to DataFrame (if not already in that format)
#         patient_df = pd.DataFrame([patient_data])

#         # Fetch and convert doctor data to DataFrame
#         doctor_data = self.fetch_doctor_data()
#         doctor_df = pd.DataFrame(doctor_data)

#         avg_budget = self.patients_df.loc[patient_index, 'budget_max_scaled']



#         # avg_budget = self.patients_df.loc[patient_index, 'budget_max_scaled']
        
#         # # Filter doctors based on the average budget (scaled) of similar patients
#         # suitable_doctors = self.doctors_df[self.doctors_df['cost_max_scaled'] <= avg_budget]
        
#         # # Compute similarity between the patient preferences and doctors using self.patient_features
#         # doctor_similarity = cosine_similarity([self.patient_features[patient_index]], suitable_doctors[['sex_encoded', 'location_encoded', 'cost_max_scaled']].to_numpy())
        
#         # # Find top N indices of doctors based on similarity scores
#         # top_doctor_indices = np.argsort(-doctor_similarity[0])[:top_n]
#         # recommended_doctors = suitable_doctors.iloc[top_doctor_indices]

#         # # patient_data = self.fetch_patient_data_by_id(patient_id)
#         # # doctor_data = self.fetch_doctor_data()


        
#         # return recommended_doctors[['doctor_id','first_name', 'last_name', 'speciality', 'location', 'cost_max']]


#     def save_model(self, file_path):
#         # Save the model components to a file
#         with open(file_path, 'wb') as f:
#             pickle.dump(self, f)

#     @staticmethod
#     def load_model(file_path):
#         # Load the model components from a file
#         with open(file_path, 'rb') as f:
#             model = pickle.load(f)
#         return model

# model = DoctorRecommendationModel()

# patients_df = pd.read_csv('../data/patients_data (1).csv')
# doctors_df = pd.read_csv('../data/doctors-data.csv')

# model.fit(patients_df, doctors_df)  
# model.save_model('doctor_recommendation_model.pkl')

# loaded_model = DoctorRecommendationModel.load_model('doctor_recommendation_model.pkl')

# # Replace 0 with any valid patient index from your dataset
# patient_index = 5
# top_n = 5  # Number of doctor recommendations you want

# # Get recommendations
# recommended_doctors = loaded_model.recommend_doctors(patient_index, top_n=top_n)

# # Print the recommendations
# print(recommended_doctors)
