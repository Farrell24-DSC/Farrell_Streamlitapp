#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[2]:


class HotelBookingModel:
    def __init__(self, path):
        self.path = path
        self.model = None

    def load_data(self):  # Preprocessing
        df = pd.read_csv(self.path)
        df.drop(columns=["Booking_ID"], inplace=True)

        # Fill missing values 
        df['type_of_meal_plan'] = df['type_of_meal_plan'].fillna(df['type_of_meal_plan'].mode()[0])
        df['required_car_parking_space'] = df['required_car_parking_space'].fillna(0)
        df['avg_price_per_room'] = df['avg_price_per_room'].fillna(df['avg_price_per_room'].median())

        # Encode categorical columns
        categorical = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        le = LabelEncoder()
        for col in categorical:
            df[col] = le.fit_transform(df[col])

        # Encode target
        df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
        return df

    def train_model(self):
        df = self.load_data()
        X = df.drop('booking_status', axis=1)
        y = df['booking_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)


# In[3]:


if __name__ == "__main__":
    model = HotelBookingModel("Dataset_B_hotel.csv")
    model.load_data()
    model.train_model()


# In[ ]:




