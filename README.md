# AI and Computer Vision Projects Collection

This repository contains a unified Streamlit web application that houses a collection of Artificial Intelligence and Computer Vision mini-projects. Originally developed as standalone scripts, they have been refactored into a single, cohesive web interface for easy demonstration and usage.

## Features

The web application provides a sidebar navigation menu to switch between the following projects:

1. **Poem Generator (YOLO + Gemini API)**
   The flagship project. Users upload an image, and the app uses a custom-trained YOLO model (best_3.pt) to detect objects in the scene. The detected objects are then fed into the Gemini 2.5 Flash Lite API to generate a personalized poem mimicking the style of famous Colombian authors. The app includes a session-based usage limit to protect API quotas.

2. **Object Counter**
   A computer vision utility that uses OpenCV contour detection and thresholding to count distinct objects in an uploaded image. It includes adjustable threshold and area filters directly from the web interface.

3. **Face and Eye Detector**
   A facial recognition script utilizing OpenCV Haar Cascades. Users can take a photo with their webcam directly from the browser, and the app processes the image to highlight faces and eyes with bounding boxes.

4. **Color Filter Detector**
   An HSV color space mask implementation. Users can capture a photo or upload an image and apply different color isolation filters (Green, Red, Yellow, Blue) to see the background removal and contour detection in real-time.

## Installation and Setup

1. Clone the repository.
2. Ensure you have Python installed.
3. Install the dependencies using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.streamlit/secrets.toml` file in the root directory and add your Google Gemini API key:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

## Running the Application

To launch the web interface locally, run:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Model Information

The custom YOLO model used for the Poem Generator is located in `models/best_3.pt`. Ensure this file remains in the correct directory for the object detection to work.
