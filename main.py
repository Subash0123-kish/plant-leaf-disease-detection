from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np

# --------------------------------------------------
# Paths (based on this file's location)
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH_KERAS = BASE_DIR / "trained_model.keras"
MODEL_PATH_H5 = BASE_DIR / "trained_model.h5"

# --------------------------------------------------
# Load model once and cache it (fast in Streamlit)
# --------------------------------------------------
@st.cache_resource
def load_plant_model():
    # Always load only the .h5 model (since .keras is invalid)
    if MODEL_PATH_H5.exists():
        model = tf.keras.models.load_model(MODEL_PATH_H5, compile=False)
        return model

    # If .h5 is missing, raise a clear error
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH_H5}")

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = load_plant_model()

    # Streamlit's uploader returns a file-like object – load_img can read it
    image = tf.keras.utils.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.utils.img_to_array(image)

    # Convert single image to a batch: (1, 128, 128, 3)
    input_arr = np.expand_dims(input_arr, axis=0)

    # Get prediction
    prediction = model.predict(input_arr)
    result_index = int(np.argmax(prediction, axis=1)[0])
    return result_index


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(
        """
        Welcome to the Plant Disease Recognition System! 🌿🔍
        ...
        """
    
        """
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """
    )

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown(
        """
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. 
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves 
    categorized into 38 different classes. The total dataset is divided into an 80/20 
    ratio of training and validation set preserving the directory structure. A new 
    directory containing 33 test images is created later for prediction purpose.

    #### Content
    1. Train (70,295 images)  
    2. Valid (17,572 images)  
    3. Test (33 images)
    """
    )

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict Button
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                st.write("Our Prediction")

                result_index = model_prediction(test_image)

                # Class names
                class_name = [
                    "Apple___Apple_scab",
                    "Apple___Black_rot",
                    "Apple___Cedar_apple_rust",
                    "Apple___healthy",
                    "Blueberry___healthy",
                    "Cherry_(including_sour)___Powdery_mildew",
                    "Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)___Common_rust_",
                    "Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy",
                    "Grape___Black_rot",
                    "Grape___Esca_(Black_Measles)",
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Grape___healthy",
                    "Orange___Haunglongbing_(Citrus_greening)",
                    "Peach___Bacterial_spot",
                    "Peach___healthy",
                    "Pepper,_bell___Bacterial_spot",
                    "Pepper,_bell___healthy",
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy",
                    "Raspberry___healthy",
                    "Soybean___healthy",
                    "Squash___Powdery_mildew",
                    "Strawberry___Leaf_scorch",
                    "Strawberry___healthy",
                    "Tomato___Bacterial_spot",
                    "Tomato___Early_blight",
                    "Tomato___Late_blight",
                    "Tomato___Leaf_Mold",
                    "Tomato___Septoria_leaf_spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy",
                ]

                if 0 <= result_index < len(class_name):
                    st.success(f"Model is predicting it's **{class_name[result_index]}**")
                else:
                    st.error("Prediction index out of range. Please check the model or class list.")
    else:
        st.info("Please upload an image to start prediction.")