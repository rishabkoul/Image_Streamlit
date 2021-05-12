# Core pkgs
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

from tensorflow.keras.models import load_model
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loadmodel():
    model = load_model('model_vgg19.h5')
    return model


def import_and_predict(image_data, model):
    vechicle = ['bike', 'boat', 'bus', 'car', 'cycle',
                'helicopter', 'plane', 'scooty', 'train', 'truck']
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    img_reshape = tf.cast(img_reshape, tf.float32)
    preds = model.predict(img_reshape)
    top3 = sorted(range(len(preds[0])), key=lambda i: preds[0][i])[-3:]
    top3.reverse()

    sentence = ""
    for i in top3:
        sentence = sentence+str(vechicle[i])+": %"+str(preds[0][i]*100)+"\n"

    return sentence


def main():
    model = loadmodel()
    """Vechicle Identification App"""

    st.title("Vechicle Identification App")

    activities = ['Identification', 'About']
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Identification':
        st.subheader("Classify Vechicles")
        st.subheader(
            "This app can identify bike, boat, bus, car, cycle,helicopter, plane, scooty, train and truck")
        st.write("For more info see about section")

        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.image(our_image)
            text = import_and_predict(our_image, model)
            st.write(text)

    elif choice == 'About':
        st.subheader('About')
        st.write(
            'This app was made using the vechicle dataset on Kaggle [https://www.kaggle.com/rishabkoul1/vechicle-dataset](https://www.kaggle.com/rishabkoul1/vechicle-dataset)')
        st.write(
            'Code of this app - [https://github.com/rishabkoul/ImageClassifier](https://github.com/rishabkoul/ImageClassifier)')
        st.write(
            'Code of model building - [https://www.kaggle.com/rishabkoul1/80-val-accuracy-with-vgg16-transfer-learning](https://www.kaggle.com/rishabkoul1/80-val-accuracy-with-vgg16-transfer-learning)')


if __name__ == "__main__":
    main()
