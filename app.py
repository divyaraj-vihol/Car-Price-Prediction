import streamlit as st
import pandas as pd
import joblib
import numpy as np
st.set_page_config(page_title="DV Prediction")
                   

# Load model and encoders
model = joblib.load("car_price_model.pkl")
model_features = joblib.load("model_features.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Set app layout
st.image('https://www.pixelstalk.net/wp-content/uploads/2016/06/Lamborghini-Desktop-HD-Car-Wallpapers.jpg',
         use_container_width=True)
st.title("🚗 Premium Car Price Predictor")

def safe_encode(encoder, value):
    """Safely encode values handling unseen labels"""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # Special value for unknown categories

def user_input_features():
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        manufacturer = st.selectbox("Manufacturer", label_encoders['Manufacturer'].classes_)
        model_name = st.text_input("Model", "Camry")
        prod_year = st.slider("Production Year", 1990, 2025, 2020)
        category = st.selectbox("Category", label_encoders['Category'].classes_)
        
    with col2:
        leather = st.selectbox("Leather Interior", ["Yes", "No"])
        fuel_type = st.selectbox("Fuel Type", label_encoders['Fuel_type'].classes_)
        engine_vol = st.number_input("Engine Volume (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
        
    with col3:
        cylinders = st.number_input("Cylinders", min_value=0, max_value=16, value=4)
        gearbox = st.selectbox("Gear Box Type", label_encoders['Gear_box_type'].classes_)
        drive_wheels = st.selectbox("Drive Wheels", label_encoders['Drive_wheels'].classes_)
        doors = st.selectbox("Doors", ["02", "04", "06", "08"])
    
    col4, col5 = st.columns(2)
    
    with col4:
        wheel = st.selectbox("Wheel", ["Left wheel", "Right wheel"])
        color = st.selectbox("Color", label_encoders['Color'].classes_)
        
    with col5:
        airbags = st.number_input("Airbags", min_value=0, max_value=16, value=6)
        levy = st.number_input("Levy", min_value=0, value=0)
    
    # Calculate additional features
    car_age = 2025 - prod_year
    
    # Prepare data with EXACT column names
    data = {
        'ID': 0,  # Default ID
        'Levy': levy,
        'Manufacturer': manufacturer,
        'Model': model_name,
        'Prod._year': prod_year,
        'Category': category,
        'Leather_interior': leather,
        'Fuel_type': fuel_type,
        'Engine_volume': engine_vol,
        'Mileage': mileage,
        'Cylinders': cylinders,
        'Gear_box_type': gearbox,
        'Drive_wheels': drive_wheels,
        'Doors': doors,
        'Wheel': wheel,
        'Color': color,
        'Airbags': airbags,
        'Car_Age': car_age
    }
    
    return pd.DataFrame([data])

# Main app logic
input_df = user_input_features()

# Display input
st.subheader("Your Car Specifications")
st.table(input_df[model_features].T.rename(columns={0: "Value"}))

if st.button("Predict Price"):
    try:
        # Create a copy for encoding
        encoded_df = input_df.copy()
        
        # Encode categorical variables with safety checks
        for feature in label_encoders:
            if feature in encoded_df.columns:
                encoded_df[feature] = encoded_df[feature].apply(
                    lambda x: safe_encode(label_encoders[feature], x)
                )
        
        # Ensure all model features are present with correct data types
        for feature in model_features:
            if feature not in encoded_df.columns:
                encoded_df[feature] = 0  # Default value
            if feature in label_encoders:
                encoded_df[feature] = encoded_df[feature].astype(int)
        
        # Reorder columns to exactly match model expectations
        encoded_df = encoded_df[model_features]
        
        # Debug check
        st.write("Final Features Being Sent to Model:")
        st.write(encoded_df)
        
        # Make prediction
        prediction = model.predict(encoded_df)
        st.success(f"Estimated Price: ${prediction[0]:,.2f}")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Debug Info:")
        st.write(f"Input features: {input_df.columns.tolist()}")
        st.write(f"Model expects: {model_features}")
        st.write(f"Encoded features: {encoded_df.columns.tolist() if 'encoded_df' in locals() else 'Not available'}")