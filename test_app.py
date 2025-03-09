import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask

# Test imports from our application
try:
    from fitness_analyzer import FitnessAnalyzer
    from plan_generator import PlanGenerator
    from image_processor import ImageProcessor
    print("✅ Successfully imported application modules")
except Exception as e:
    print(f"❌ Error importing application modules: {e}")
    sys.exit(1)

# Test creating instances of our classes
try:
    fitness_analyzer = FitnessAnalyzer()
    plan_generator = PlanGenerator()
    image_processor = ImageProcessor()
    print("✅ Successfully created class instances")
except Exception as e:
    print(f"❌ Error creating class instances: {e}")
    sys.exit(1)

# Test Flask app creation
try:
    app = Flask(__name__)
    print("✅ Successfully created Flask app")
except Exception as e:
    print(f"❌ Error creating Flask app: {e}")
    sys.exit(1)

# Print versions
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

print("\nAll tests completed successfully! The application should be ready to run.")
print("You can start the application with: python app.py") 