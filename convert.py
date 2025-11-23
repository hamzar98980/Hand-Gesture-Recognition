from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_SIZE = (64, 64)

def convert_to_dataset_style(uploaded_file):
    # 1) Read image
    img = Image.open(uploaded_file).convert("RGB")
    
    # 2) Convert to grayscale
    img_gray = img.convert("L")

    # 3) Resize
    img_gray = img_gray.resize(IMG_SIZE)

    # 4) Convert to numpy
    arr = np.array(img_gray)

    # 5) Otsu threshold → binary silhouette
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6) Normalize
    binary = binary / 255.0

    # 7) Flatten
    flat = binary.reshape(1, -1)

    return img_gray, binary, flat

# ---- FIXED PATH (use raw string r"..." ) ----
gray_img, binary_img, flat = convert_to_dataset_style(
    r"C:\laragon\www\hackathon-smit\custome2.jpg"
)

# ---- Show images ----
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale 64×64")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(binary_img, cmap="gray")
plt.title("Binary Silhouette (Dataset Style)")
plt.axis("off")

plt.show()
