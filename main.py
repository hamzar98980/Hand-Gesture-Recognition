import joblib
import streamlit as slt
from PIL import Image
import numpy as np

# ---- Load model once ----
lr = joblib.load("model/logistic_regression.pkl")



slt.title('Hand Gesture Recognition')
upload_file = slt.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])


IMG_SIZE = (64, 64)
CLASS_NAMES = ['0', '1', '10', '11', '12', '13', '14', '15', '16',
               '17', '18', '19', '2', '3', '4', '5', '6', '7', '8', '9']

def preprocess_like_training(file_obj):
    # file_obj is the uploaded file from Streamlit
    img = Image.open(file_obj).convert("L")   # grayscale

    # resize to 64x64 like training data
    img = img.resize(IMG_SIZE)

    # to numpy + normalize
    arr = np.array(img).astype("float32") / 255.0      # (64, 64)

    # flatten to 4096 features
    flat = arr.reshape(1, -1)                          # (1, 4096)

    # predict
    pred_idx = lr.predict(flat)[0]                     # int label
    label_name = CLASS_NAMES[pred_idx]

    return flat, arr, label_name, pred_idx

if upload_file is not None:
    # show original image
    slt.image(upload_file, width=150, caption="Uploaded image")

    flat, arr, label_name, pred_idx = preprocess_like_training(upload_file)

    # slt.write(f"**Predicted class index:** {pred_idx}")
    slt.title(f'Prediction: {label_name}')

    # (Optional) also show the preprocessed 64x64 grayscale image
    # slt.image(arr, width=150, caption="Preprocessed (64x64 grayscale)")


slt.divider()
slt.title("Model Evaluation")

slt.header("Sample Dataset")
sampledataset_img = Image.open("sample_dataset.png")
slt.image(sampledataset_img, caption="Sample Dataset",width=500)

slt.header("Confusion Matrix")
cm_img = Image.open("confusion_matrix.png")
slt.image(cm_img, caption="Confusion Matrix",width=500)

slt.header("Per Class Accuracy")
sampledataset_img = Image.open("perr_class_accuracy.png")
slt.image(sampledataset_img, caption="Per Class Accuracy",width=500)
