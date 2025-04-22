#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
import pickle


# In[9]:


class HotelModelPredictor:
    def __init__(self, model_path="UTS_Model/best_model.pkl"):
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
        # Imputasi manual sesuai training
        if pd.isna(input_df['type_of_meal_plan'].values[0]):
            input_df['type_of_meal_plan'] = 'Meal Plan 1'
        if pd.isna(input_df['avg_price_per_room'].values[0]):
            input_df['avg_price_per_room'] = 99.45
        if pd.isna(input_df['required_car_parking_space'].values[0]):
            input_df['required_car_parking_space'] = 0

        # Manual encoding pakai map
        for col in self.label_columns:
            input_df[col] = input_df[col].map(self.encoder_maps[col])

        return input_df

    def predict(self, input_data):
        processed = self.preprocess_input(input_data)
        prediction = self.model.predict(processed)[0]
        return "Canceled" if prediction == 1 else "Not_Canceled"


# In[10]:


predictor = HotelModelPredictor()

# UI Streamlit
st.title('Hotel Booking Cancellation Predictor')
st.sidebar.header("Input Your Booking Information")
input_data = {
    'no_of_adults': st.sidebar.number_input('Number of Adults', min_value=0, value=2),
    'no_of_children': st.sidebar.number_input('Number of Childrens', min_value=0, value=0),
    'no_of_weekend_nights': st.sidebar.number_input('Number of Weekend Nights', min_value=0, value=1),
    'no_of_week_nights': st.sidebar.number_input('Number of Week Nights', min_value=0, value=2),
    'type_of_meal_plan': st.sidebar.selectbox('Meal Plans (1,2,3)', ['Meal Plan 1', 'Meal Plan 2', 'Not Selected', 'Meal Plan 3']),
    'required_car_parking_space': st.sidebar.selectbox('Car Park Needed', [0, 1]),
    'room_type_reserved': st.sidebar.selectbox('Room Type[1-7]', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
    'lead_time': st.sidebar.number_input('Lead Time (in days)', min_value=0, value=15),
    'arrival_year': st.sidebar.number_input('Arrival Year', min_value=2020, max_value=2030, value=2025),
    'arrival_month': st.sidebar.selectbox('Arrival Month', list(range(1, 13))),
    'arrival_date': st.sidebar.number_input('Arrival Date', min_value=1, max_value=31, value=15),
    'market_segment_type': st.sidebar.selectbox('Market Segment', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation']),
    'repeated_guest': st.sidebar.selectbox('Repeated Guest', [0, 1]),
    'no_of_previous_cancellations': st.sidebar.number_input('Previous Cancellations', min_value=0, value=0),
    'no_of_previous_bookings_not_canceled': st.sidebar.number_input('Previous Non-canceled Bookings', min_value=0, value=1),
    'avg_price_per_room': st.sidebar.number_input('Avg Price per Room', min_value=0.0, value=100.0),
    'no_of_special_requests': st.sidebar.number_input('Special Requests', min_value=0, value=0),
}


# In[11]:


if st.button('Predict'):
    result = predictor.predict(input_data)
    st.write(f"Prediction: {result}")


# In[ ]:




