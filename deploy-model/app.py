#===============================================================================================#

# Imports
# http://skincare-rating-prediction.herokuapp.com/

#===============================================================================================#

import streamlit as st

import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#===============================================================================================#

# Functions and Models Prepared

#===============================================================================================#

tokenizer = joblib.load(open('model/save_tokenizer.pkl', 'rb'))
neural_net_model = load_model('model/base_model_imp.h5')


#===============================================================================================#

# Streamlit

#===============================================================================================#

st.title("Skincare Review Rating Classifier")

review_text = st.text_area('Enter Your Review Here')

if st.button('Predict'):
    
    result_review = review_text.title()
    review_text = tokenizer.texts_to_sequences([result_review])
    review_text = pad_sequences(review_text, padding='post',maxlen=250)


    prediction = neural_net_model.predict(review_text)
    
    prediction = np.argmax(prediction)
    
    st.success(prediction+2)