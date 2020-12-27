import tensorflow as tf
model = tf.keras.models.load_model('flower_classification.h5')
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np


st.write("""
         # Flower Classification App
         """
         )
# st.write("This is a simple image classification web app to predict type of flower")

file = st.file_uploader("Please upload an image file", type=["jpg", "png",'jpeg'])



def import_and_predict(image_data, model):
    
        size = (200,200)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(150, 150),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]

        prediction = model.predict(img_reshape)[0]
        arg = np.argmax(prediction)
        dict_ = {0:"daisy",1:"dandelion",2:"rose",3:"sunflower",4:"tulip"}
        print(dict_[arg])
        
        return dict_[arg]


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    st.write(f'# {prediction}')