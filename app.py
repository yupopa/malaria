from fastai.vision.widgets import *
from fastai.vision.all import *
from pathlib import Path
import streamlit as st
from urllib.request import urlretrieve
url = 'http://dl.dropboxusercontent.com/s/kcmjcwxjtxnpt5n/malaria.pkl?raw=1'
filename = 'malaria.pkl'
urlretrieve(url,filename)
st.markdown("PARASITIZED OR UNINFECTED(MALARIA)")
st.write("This app classifies parasitized cells or uninfected cells(malaria)")
st.write("This model trained using these images ->  https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria")


class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to learn parasitized or uninfected') 

if __name__=='__main__':
    predictor = Predict(filename)
