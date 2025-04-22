#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pickle
import numpy as np
import pandas as pd


# In[18]:


class HotelModelPredictor:
    def __init__(self, model_path="best_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Mapping encoding langsung
        self.encoder_maps = {
            'type_of_meal_plan': {
                'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3
            },
            'room_type_reserved': {
                'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3,
                'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7
            },
            'market_segment_type': {
                'Online': 1, 'Offline': 2, 'Corporate': 3, 'Aviation': 4, 'Complementary': 5
            }
        }

        self.label_columns = list(self.encoder_maps.keys())


    def preprocess_input(self, input_data):
        input_df = pd.DataFrame([input_data])
        #Imputasi manual training
        if pd.isna(input_df['type_of_meal_plan'].values[0]):
            input_df['type_of_meal_plan'] = 'Meal Plan 1'
            
        if pd.isna(input_df['avg_price_per_room'].values[0]):
            input_df['avg_price_per_room'] = 99.45  
            
        if pd.isna(input_df['required_car_parking_space'].values[0]):
            input_df['required_car_parking_space'] = 0  

        #Encoding Kategori
        for col in self.label_columns:
            input_df[col] = input_df[col].map(self.encoder_maps[col])

        return input_df

    def predict(self, input_data):
        processed = self.preprocess_input(input_data)
        prediction = self.model.predict(processed)[0]
        return "Canceled" if prediction == 1 else "Not_Canceled"


# ## Contoh Kasus

# In[19]:


predictor = HotelModelPredictor()

sample_input = {
    'no_of_adults': 2,
    'no_of_children': 0,
    'no_of_weekend_nights': 2,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 1',
    'required_car_parking_space': 0,
    'room_type_reserved': 'Room_Type 1',
    'lead_time': 45,
    'arrival_year': 2017,
    'arrival_month': 5,
    'arrival_date': 12,
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 120.0,
    'no_of_special_requests': 1
}


result = predictor.predict(sample_input)
print("Hasil prediksi:", result)


# In[ ]:




