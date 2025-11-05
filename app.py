import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tensorflow.keras.backend as K

# --- 1. Define the Custom Functions (REQUIRED) ---
# The model will not load without these exact functions.

@st.cache_data # Use cache_data for functions
def iou(y_true, y_pred, smooth=1e-6):
    """Intersection over Union (IoU) metric."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Threshold the prediction
    y_pred_f = K.cast(y_pred_f > 0.5, y_true_f.dtype)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    return (intersection + smooth) / (union + smooth)

@st.cache_data # Use cache_data for functions
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice Loss function."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    return 1 - ( (2. * intersection + smooth) / 
                 (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) )


# --- 2. Load the Model (Cached) ---
# Caching speeds up the app, loading the model only once.

@st.cache_resource # Use cache_resource for large objects like models
def load_app_model():
    """Loads the saved Keras model."""
    model_path = 'cholecseg8k_unet_final.keras'
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'dice_loss': dice_loss, 'iou': iou}
    )
    return model

model = load_app_model()


# --- 3. Helper Functions for Processing ---

def preprocess_image(image_pil, target_size=(256, 256)):
    """
    Prepares a PIL image for the model.
    1. Converts to RGB numpy array
    2. Resizes
    3. Normalizes
    4. Adds batch dimension
    """
    # Convert PIL Image to NumPy array
    image_np = np.array(image_pil.convert('RGB'))
    
    # Resize to the model's expected input size
    image_resized = cv2.resize(image_np, target_size)
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized / 255.0
    
    # Add batch dimension (1, H, W, C)
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Return the processed batch and the display-friendly resized image
    return image_batch, image_resized

def predict_mask(model_to_predict, image_batch):
    """
    Runs model prediction and post-processes the mask.
    """
    # Predict the mask
    pred_mask_prob = model_to_predict.predict(image_batch)
    
    # Remove batch dimension: (1, H, W, 1) -> (H, W, 1)
    pred_mask_prob = pred_mask_prob[0]
    
    # Threshold probabilities at 0.5 to get binary mask [0 or 1]
    pred_mask_binary = (pred_mask_prob > 0.5).astype(np.uint8)
    
    return pred_mask_binary


# --- 4. Streamlit Web Interface ---

st.title("🩺 Surgical Tool Segmentation")
st.write("Upload a surgical image and the U-Net model will predict the segmentation mask.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load the uploaded image
    image = Image.open(uploaded_file)
    
    # 2. Pre-process the image for the model
    image_batch, display_image = preprocess_image(image)
    
    # 3. Get the model's prediction
    # Add a spinner while the model is working
    with st.spinner('Running segmentation...'):
        predicted_mask = predict_mask(model, image_batch)
    
    # 4. Display the results
    st.write("### Model Results")
    
    # Use columns for a side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(display_image, caption="Original Image (Resized to 256x256)")
    
    with col2:
        # Squeeze the mask to (H, W) for display
        st.image(predicted_mask.squeeze(), caption="Predicted Mask", cmap='gray')