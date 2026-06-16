#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Load the previously saved model and scaler
@st.cache_resource
def load_assets():
    model = load_model('cancer_cnn_model_V3.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("Lung Cancer VOC Prediction")

# 27 Chemical Markers extracted from the dataset
markers = [
    'Formaldehyde(CH2O)', 'Acetaldehyde(C2H4O)', 'Acetone(C3H6O)', 'Butyraldehyde(C4H8O)', 
    '2-Pentanone(C5H10O)', 'Hexanal(C6H12O)', 'Heptaldehyde(C7H14O)', 'Caprylaldehyde(C8H16O)', 
    '1-Nonen-3-ol(C9H18O)', 'Menthol(C10H20O)', 'Undecylaldehyde(C11H22O)', 'Laurinaldehyde(C12H24O)', 
    '2-Tridecanone(C13H26O)', 'Butyric acid(C4HO8O2)', 'Acetic acid(C2H4O2)', 'Propargyl alcohol(C3H4O)', 
    'Acetonylacetone(C6H10O2)', 'Azelaic acid(C9H16O2)', 'Acrylic acid(C3H4O2)', 'Crotonic acid(C4H6O2)', 
    'Crotonaldehyde(C4H6O)', 'Fumaraldehyde(C4H4O2)', 'Cyclopentanone(C5H8O)', 'Benzaldehyde(C7H6O)', 
    'Cyclohexylmethanone(C7H11O)', '1-Tridecanol(C13H22O)', '9-Anthraldehyde(C15H10O)'
]

st.subheader("Enter VOC Marker Levels")
inputs = []
cols = st.columns(3)

# 1. Typed inputs with empty defaulting to 0
for i, marker in enumerate(markers):
    with cols[i % 3]:
        val = st.text_input(marker, value="")
        inputs.append(float(val) if val.strip() != "" else 0.0)

if st.button("Analyze"):
    input_array = np.array([inputs])
    scaled_input = scaler.transform(input_array)
    reshaped_input = scaled_input.reshape((1, 27, 1))

    # Generate prediction
    confidence = float(model.predict(reshaped_input)[0][0])

    st.markdown("---")

    # Analysis 1: Confidence Level
    st.subheader("1. Confidence Level")
    st.progress(confidence)
    st.metric(label="Cancer Probability", value=f"{confidence * 100:.2f}%")

    st.markdown("---")

    # Analysis 2: 27 Chemical Spikes
    st.subheader("2. Chemical Spikes")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(markers, inputs, color='coral')
    plt.xticks(rotation=90)
    plt.ylabel("Input Value")
    st.pyplot(fig)

    st.markdown("---")

    # Analysis 3: Contributing Spikes (Flags)
    st.subheader("3. Model Triggers")
    if confidence >= 0.5:
        # Identifies the highest standard deviations above the training mean
        top_indices = np.argsort(scaled_input[0])[-3:][::-1]
        culprits = [markers[i] for i in top_indices]
        st.error(f"**Flagged for Cancer.** The top 3 chemical spikes contributing to this result are: **{', '.join(culprits)}**")
    else:
        st.success("No cancer detected. Marker levels are within normal bounds.")


# In[ ]:




