# Fitness Analyzer

A computer vision-based fitness analysis application that:
1. Analyzes uploaded images to identify fitness equipment and activities
2. Analyzes body images to determine body type, proportions, and composition
3. Collects user data (height, weight, fitness goals, etc.)
4. Generates personalized exercise and diet plans based on image analysis and user data

## Features
- Image recognition for fitness equipment and activities
- Body analysis using pose estimation to determine body type and proportions
- User profile management
- Personalized exercise plan generation based on available equipment and body analysis
- Diet recommendation system tailored to body type and fitness goals
- Nutrient recommendations based on body composition

## Setup
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the application:
```
python app.py
```

3. Access the web interface at http://localhost:5000

## How It Works
1. **User Profile**: Enter your personal information, fitness goals, and dietary preferences
2. **Image Upload**: Upload either:
   - Images of your fitness equipment for equipment detection
   - A full-body image of yourself for body analysis
3. **AI Analysis**: Our system analyzes the images using:
   - OpenCV and TensorFlow for equipment recognition
   - MediaPipe Pose for body analysis and measurements
4. **Personalized Plans**: Receive customized exercise and diet plans based on:
   - Available equipment
   - Body type and composition
   - Fitness goals
   - Dietary preferences

## Project Structure
- `app.py`: Main Flask application
- `image_processor.py`: OpenCV-based image processing module with body analysis
- `fitness_analyzer.py`: Fitness equipment/activity recognition and body status analysis
- `plan_generator.py`: Exercise and diet plan generation based on all available data
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and static assets
- `models/`: Pre-trained models for image recognition
- `data/`: Sample data and reference information 