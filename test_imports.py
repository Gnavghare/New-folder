"""
Test script to verify that all required imports are working correctly.
"""

# Test basic imports
print("Testing basic imports...")
try:
    import numpy as np
    import cv2
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print("✓ Basic imports successful")
except ImportError as e:
    print(f"✗ Error importing basic packages: {e}")

# Test MediaPipe imports
print("\nTesting MediaPipe imports...")
try:
    import mediapipe as mp
    print("✓ MediaPipe imports successful")
except ImportError as e:
    print(f"✗ Error importing MediaPipe: {e}")

# Test Flask imports
print("\nTesting Flask imports...")
try:
    from flask import Flask, render_template, request, redirect, url_for
    from flask_wtf import FlaskForm
    from flask_sqlalchemy import SQLAlchemy
    print("✓ Flask imports successful")
except ImportError as e:
    print(f"✗ Error importing Flask packages: {e}")

# Print TensorFlow and OpenCV versions
print(f"\nTensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

print("\nAll import tests completed.") 