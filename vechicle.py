# Core pkgs
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

from tensorflow.keras.models import load_model

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
    preds = model.predict(img_reshape)
    top3 = sorted(range(len(preds[0])), key=lambda i: preds[0][i])[-3:]
    top3.reverse()

    sentence = ""
    for i in top3:
        sentence = sentence+str(vechicle[i])+": %"+str(preds[0][i]*100)+"\n"

    return sentence


def main():
    model = loadmodel()
    """Image Classification App"""

    st.title("Image Classification App")

    activities = ['Classification', 'About']
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Classification':
        st.subheader("Classify Vechicles")

        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.image(our_image)
            text = import_and_predict(our_image, model)
            st.write(text)

    elif choice == 'About':
        st.subheader('About')
        st.write('This app was made by Rishab Koul with streamlit')


if __name__ == "__main__":
    main()
