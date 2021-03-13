from fastai.vision.widgets import *
from fastai.vision.all import *
from pathlib import Path
import streamlit as st
from urllib.request import urlretrieve
urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "fig-03_5.png"
urlretrieve(urll,filenamee)
st.image(filenamee)


url = 'http://dl.dropboxusercontent.com/s/kcmjcwxjtxnpt5n/malaria.pkl?raw=1'
filename = 'malaria.pkl'
urlretrieve(url,filename)
st.markdown("Detection of malaria disease through prediction from red blood cell images")
st.write("This app is used to predict the infectious condition of the red blood cells as healthy or infected.")
st.write("The following dataset for image classification was used to train the model     https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria")



class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
 
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')
        pred, pred_idx, probs = self.learn_inference.predict(self.img)
        st.write(f'Prediction: {pred} red blood cell; Probability: {probs[pred_idx]:.04f}')

 

if __name__=='__main__':
    predictor = Predict(filename)
