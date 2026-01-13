🍔 Food Recognition & Calorie Estimation System

A real-time computer vision application that detects food items using a webcam and estimates approximate calories per serving using a pretrained deep learning model.

📌 Overview

This project uses MobileNetV2 (ImageNet) for food recognition and maps detected food items to a curated calorie database to provide quick nutritional estimates. The system runs in real time using a webcam and OpenCV.

It is designed as a practical demonstration of deep learning, computer vision, and model deployment.

🚀 Features

Real-time food detection using webcam

Top-5 prediction display with confidence scores

Calorie estimation per serving

Lightweight and fast inference using MobileNetV2

Smart label-to-food mapping for calorie lookup

Simple keyboard-based interaction

🧠 Technology Stack

Python

TensorFlow / Keras

MobileNetV2 (pretrained on ImageNet)

OpenCV

NumPy

Pillow

⚙️ How It Works

Captures an image from the webcam

Preprocesses the frame and feeds it to MobileNetV2

Extracts top-5 predictions from ImageNet

Maps detected food labels to a calorie database

Displays food name and estimated calories per serving

🕹 Controls

SPACE → Capture frame and classify food

ESC → Exit application
