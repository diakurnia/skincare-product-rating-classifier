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
neural_net_model = load_model('model/base_model.h5')


st.title("What's your opinion about the product?")
st.title("Label hasil prediksi pada aplikasi ini hanya ada 4 yaitu (2,3,4,5)")
review_text = st.text_area('Enter Your Review Here')

if st.button('Predict'):
    
    result_review = review_text.title()
    review_text = tokenizer.texts_to_sequences([result_review])
    review_text = pad_sequences(review_text, padding='post', maxlen=30)


    prediction_proba = neural_net_model.predict(review_text)
    # print(prediction)
    
    prediction = np.argmax(prediction_proba)
    
    st.write("Hasil prediksi berdasarkan review diatas adalah")
    st.write( prediction + 2)
    