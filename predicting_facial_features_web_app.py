import os

import PIL
import requests
import streamlit as st
import tensorflow as tf

from image_preprocessor import image_preprocessor

# streamlit page setup

st.set_page_config(
    page_title="Predicting facial features",
    page_icon=":man:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# facial features prediction model setup


@st.cache_resource
def loading_ml_model():
    model_directory = "Model"
    path_to_model = os.path.join(os.getcwd(), model_directory)
    model_to_load = os.listdir(path_to_model)[0]
    model = tf.keras.models.load_model(os.path.join(path_to_model, model_to_load))
    return model


loaded_model = loading_ml_model()

# facial features names setup

features = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips',
            'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows', 'Chubby',
            'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones', 'Male',
            'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose',
            'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair',
            'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie', 'Young']

_, center, _ = st.columns([1, 5, 1])  # is used to "cheat" center alignment of elements
center.title(":red[Predicting facial features]")

st.header("Upload images: ")

uploaded_images = st.file_uploader(label="uploaded_images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, label_visibility="hidden")

_, center, _ = st.columns([1, 2.85, 1])
if center.button(label="Or use an example: a thispersondoesnotexist.com image"):
    url = "https://thispersondoesnotexist.com/"
    image = requests.get(url).content
    st.header(":violet[Example preprocess results and predictions: ]")
    with st.expander(label="thispersondoesnotexist.com/image"):
        col1, col2 = st.columns(2)
        try:
            col1.image(image=image, caption="Original image")
            output_image = image_preprocessor(image)
        except PIL.UnidentifiedImageError:
            st.error("Could not load image from [thispersondoesnotexist.com/image](https://thispersondoesnotexist.com/image) (webserver could be down)")
        else:
            if output_image is not None:
                col2.image(image=output_image, caption="Cropped and aligned")

                image_tensor = tf.image.convert_image_dtype(output_image, dtype=tf.uint8)
                image_tensor = tf.expand_dims(image_tensor, 0)

                predictions = []

                model_predictions = loaded_model(image_tensor).numpy()[0]

                for index, prediction in enumerate(model_predictions):
                    predictions.append([features[index], [bool(round(prediction)), round(prediction, 3)]])

                for i in range(0, len(predictions), 2):  # needs to use pairs of predictions to fill columns nicely
                    parameter1, result1 = predictions[i]
                    parameter2, result2 = predictions[i + 1]

                    col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

                    if result1[0]:
                        col1.write(f"**{parameter1}** :heavy_check_mark:")
                        col2.write(f"**{result1[0]} ({result1[1]:.3f})**")
                    else:
                        col1.write(f"{parameter1} :x:")
                        col2.write(f"{result1[0]} ({result1[1]:.3f})")
                    if result2[0]:
                        col3.write(f"**{parameter2}** :heavy_check_mark:")
                        col4.write(f"**{result2[0]} ({result2[1]:.3f})**")
                    else:
                        col3.write(f"{parameter2} :x:")
                        col4.write(f"{result2[0]} ({result2[1]:.3f})")
            else:
                st.error("No face detected in the picture")
    _, center, _ = st.columns([2, 1, 2])
    center.button(label="Close example")

else:
    images_to_predict = []

    if uploaded_images:
        st.header(":blue[Preprocess results: ]")
        for image in uploaded_images:
            with st.expander(label=image.name):
                col1, col2 = st.columns(2)
                col1.image(image=image, caption="Original image")
                output_image = image_preprocessor(image.getvalue())
                if output_image is not None:
                    col2.image(image=output_image, caption="Cropped and aligned")
                    _, center, _ = st.columns([1, 2, 1])
                    try:
                        if center.checkbox(label="Predict facial features on this image", key=image.name):
                            images_to_predict.append([output_image, image.name])
                    except st.errors.DuplicateWidgetID:
                        st.error("Please don't upload duplicate images or images with same filename. ")
                else:
                    st.error("No face detected in the picture")

    @st.cache_data
    def load_model_predictions():
        loaded_model_predictions = loaded_model(image_tensor).numpy()[0]
        return loaded_model_predictions

    if images_to_predict:
        st.header(":orange[Predictions: ]")
        for image, filename in images_to_predict:
            with st.expander(label=filename):
                image_tensor = tf.image.convert_image_dtype(image, dtype=tf.uint8)
                image_tensor = tf.expand_dims(image_tensor, 0)

                predictions = []

                model_predictions = load_model_predictions()

                for index, prediction in enumerate(model_predictions):
                    predictions.append([features[index], [bool(round(prediction)), round(prediction, 3)]])

                _, center, _ = st.columns([1, 1, 1])
                center.image(image=image, caption=f"{filename[:-4]}")

                for i in range(0, len(predictions), 2):
                    parameter1, result1 = predictions[i]
                    parameter2, result2 = predictions[i + 1]

                    col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

                    if result1[0]:
                        col1.write(f":green[**{parameter1}** :heavy_check_mark:]")
                        col2.write(f"**{result1[0]} ({result1[1]:.3f})**")
                    else:
                        col1.write(f":red[{parameter1} :x:]")
                        col2.write(f"{result1[0]} ({result1[1]:.3f})")
                    if result2[0]:
                        col3.write(f":green[**{parameter2}** :heavy_check_mark:]")
                        col4.write(f"**{result2[0]} ({result2[1]:.3f})**")
                    else:
                        col3.write(f":red[{parameter2} :x:]")
                        col4.write(f"{result2[0]} ({result2[1]:.3f})")
