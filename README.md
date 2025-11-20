# 🩺 Surgical Image Analysis 
Modern minimally invasive surgeries generate large amounts of real-time video data, but manual interpretation is time-consuming and lacks quantitative precision. Surgeons often rely on visual estimation when assessing the size of tissues, lesions, or surgical targets, which can lead to variability and reduced accuracy. To address this challenge, this project proposes an AI-powered surgical video analysis system that automatically identifies, segments, and measures anatomical and surgical regions in laparoscopy/endoscopy videos.

The system is trained and evaluated using the CholecSeg8k surgical dataset, which consists of high-quality annotated frames extracted from cholecystectomy procedures. The video-based framework integrates multiple segmentation models and measurement pipelines, enabling a comprehensive comparison of techniques for real-world surgical assistance.


## ✨ Features

  * **Image Upload:** Upload surgical images (`.png`, `.jpg`, `.jpeg`).
  * **AI-Powered Segmentation:** Runs inference using a trained Keras/TensorFlow U-Net model.
  * **Side-by-Side View:** Displays the original image and the model's predicted mask.

-----

## 🛠️ Tech Stack

  * **Model:** TensorFlow / Keras (U-Net Architecture)
  * **Web App:** Streamlit
  * **Data Processing:** OpenCV, NumPy, Pillow
  * **Dataset:** CholecSeg8k

-----

## 1. The Architecture: U-Net

The model you built is a **U-Net**. It's one of the most successful and popular architectures for **biomedical image segmentation**. Its name comes from its "U" shape.



Here’s why it's so special:

* **Encoder (The "Down" Path):** The left side of the "U" is a standard convolutional network (like VGG or ResNet). It "sees" the image and captures *context*. As the image goes down, its size gets smaller, but its "knowledge" of *what* is in the image (e.g., "I see something metallic and long") gets deeper. This is the **"what"** part of the puzzle.
* **Decoder (The "Up" Path):** The right side of the "U" is the "expanding" path. It takes the deep, abstract knowledge from the bottom and starts to rebuild the image, upsampling it back to its original size. This is the **"where"** part of the puzzle.
* **The "Magic": Skip Connections:** This is the key U-Net innovation. The horizontal gray arrows in the diagram are "skip connections." They copy information from the early, high-resolution "Encoder" layers and send it directly to the late, high-resolution "Decoder" layers.

> **Analogy:** Imagine trying to draw a detailed map. The **Encoder** is like flying up in an airplane. You lose detail but get the big picture ("There's a river and a city"). The **Decoder** is like coming back down to draw the map. The **Skip Connections** are like looking at close-up photos you took on the way up, so you can remember exactly where the river banks and building edges are.

This allows the model to make highly precise, pixel-perfect masks while still understanding the overall context of the image.

<img width="1452" height="664" alt="Screenshot 2025-11-05 192920" src="https://github.com/user-attachments/assets/5544e5c3-a2d4-414c-9511-5945bfe41a8d" />


## 2. The "Language" of Learning: Dice Loss & IoU

This is the part we had to debug, and it's critical.

* **Why `accuracy` failed:** Your dataset is highly **imbalanced**. Most images are 95% "background" (0) and 5% "tool" (1). A dumb model could get 95% accuracy by just guessing "background" every time. It learns nothing.
* **Why `Dice Loss` and `IoU` work:** These metrics care about one thing: **overlap**.
    * **IoU (Intersection over Union):** This is your metric. It's the "percent overlap" between the *Ground Truth* mask and the *Predicted Mask*. An IoU of 0.8 (or 80%) means the model's prediction perfectly overlapped with 80% of the real mask, which is excellent. This is a much tougher and more honest grade.
    * **Dice Loss (1 - Dice Score):** This is your loss function. It's directly related to IoU. By using Dice Loss, you told the model: "Stop caring about the 95% of black pixels. I will *only* reward you for correctly overlapping with the 5% of white pixels (the tools)."



By changing your loss and metric, you forced the model to stop being lazy and focus on the small, important parts of the image.

## 3. The Data: CholecSeg8k

Your model is only as smart as the data it learned from.
* Your model has learned to identify the **specific pixel patterns** of surgical tools (like their metallic sheen, shape, and edges) from the 8,080 images it studied.
* It also learned the "context" in which they appear (e.g., they are usually inside a body cavity, surrounded by reddish tissue).
* The "white" pixels in your masks had a value of `50`. Your model learned that `50` in the mask file corresponds to `(mask > 10)` in its own logic, which it then learned to output as `1.0` (white).

In short, you didn't just train a generic "image model." You trained a highly specialized U-Net to be an expert in finding the *exact* patterns from the CholecSeg8k dataset, using a loss function (Dice) that forced it to pay attention to the tiny, critical details.

<img width="1459" height="497" alt="Screenshot 2025-11-05 193056" src="https://github.com/user-attachments/assets/11ce98fe-6c52-4e0d-9adf-d53112115660" />
<img width="1455" height="504" alt="Screenshot 2025-11-05 193109" src="https://github.com/user-attachments/assets/1c7fb346-4713-4054-bc84-2cc0cc7a46cb" />
<img width="1464" height="503" alt="Screenshot 2025-11-05 193122" src="https://github.com/user-attachments/assets/75a4b969-66bd-4f70-b942-875a8224241d" />

<img width="1222" height="541" alt="Screenshot 2025-11-05 193136" src="https://github.com/user-attachments/assets/23e822a5-968a-4009-a739-d9bae36cd8a8" />


## 🖥️ How to Run Locally

To run this application on your local machine, follow these steps.

### Prerequisites

You must have **Git** and **Git LFS** (Large File Storage) installed to download the large model file.

1.  [Install Git](https://www.google.com/search?q=https://git-scm.com/downloads)
2.  [Install Git LFS](https://git-lfs.github.com/)

### Step-by-Step Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SoundaryaDube/surgical-segmentation-app.git
    cd surgical-segmentation-app
    ```

2.  **Initialize Git LFS and pull the model:**

    ```bash
    git lfs install
    git lfs pull
    ```

    *(This step is crucial\! It downloads the `cholecseg8k_unet_final.keras` file that Git LFS is tracking.)*

3.  **Create a virtual environment (Recommended):**

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

Your app will now be running on your local server, and you can open it in your browser\!

-----

## 📁 Repository Structure

```
.
├── 📄 .gitattributes              # Configures Git LFS to track .keras files
├── 🐍 app.py                      # The main Streamlit application script
├── 🧠 cholecseg8k_unet_final.keras  # The trained Keras model (tracked by Git LFS)
├── 📝 requirements.txt            # Python dependencies for the app
└── 📖 README.md                   # You are here!
```

-----

## ☁️ Deployment

This app is designed to be deployed on **Streamlit Cloud**. Deployment is free and connects directly to this GitHub repository.

1.  Sign up for [Streamlit Cloud](https://streamlit.io/cloud).
2.  Connect your GitHub account.
3.  Select this repository and deploy.
