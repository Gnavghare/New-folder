import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.utils import img_to_array, load_img # type: ignore

class FitnessAnalyzer:
    def __init__(self):
        # Define body type specific exercise recommendations
        self.body_type_exercises = {
            "Inverted Triangle": {
                "focus_areas": ["lower body", "core"],
                "recommended_exercises": [
                    "squats", "lunges", "deadlifts", "glute bridges", 
                    "leg press", "hamstring curls", "calf raises",
                    "pilates", "core work"
                ],
                "avoid_exercises": [
                    "excessive upper body work", "heavy shoulder training"
                ]
            },
            "Pear": {
                "focus_areas": ["upper body", "core"],
                "recommended_exercises": [
                    "push-ups", "pull-ups", "shoulder press", "chest press",
                    "rows", "lat pulldowns", "bicep curls", "tricep extensions",
                    "HIIT training"
                ],
                "avoid_exercises": [
                    "excessive lower body isolation work"
                ]
            },
            "Athletic": {
                "focus_areas": ["balanced approach", "maintenance"],
                "recommended_exercises": [
                    "compound movements", "functional training", "sports-specific training",
                    "olympic lifts", "plyometrics", "circuit training"
                ],
                "avoid_exercises": []
            },
            "Rectangle": {
                "focus_areas": ["creating curves", "core"],
                "recommended_exercises": [
                    "shoulder work", "lat exercises", "glute training",
                    "oblique work", "waist-defining exercises",
                    "hip thrusts", "lateral raises"
                ],
                "avoid_exercises": []
            }
        }
        
        # Define BMI category specific recommendations
        self.bmi_category_recommendations = {
            "Underweight": {
                "exercise_focus": "strength training with moderate cardio",
                "exercise_intensity": "moderate",
                "exercise_frequency": "3-4 days per week",
                "cardio_recommendation": "light cardio 2-3 days per week, 20-30 minutes",
                "nutrition_focus": "caloric surplus, protein-rich foods, healthy fats",
                "meal_frequency": "5-6 smaller meals throughout the day",
                "hydration": "2-3 liters of water daily"
            },
            "Normal weight": {
                "exercise_focus": "balanced strength and cardio",
                "exercise_intensity": "moderate to high",
                "exercise_frequency": "4-5 days per week",
                "cardio_recommendation": "cardio 3-4 days per week, 30-45 minutes",
                "nutrition_focus": "balanced macronutrients, maintenance calories",
                "meal_frequency": "3-5 meals per day",
                "hydration": "2-3 liters of water daily"
            },
            "Overweight": {
                "exercise_focus": "cardio with strength training",
                "exercise_intensity": "moderate to high",
                "exercise_frequency": "5-6 days per week",
                "cardio_recommendation": "cardio 4-5 days per week, 30-60 minutes",
                "nutrition_focus": "slight caloric deficit, high protein, fiber-rich foods",
                "meal_frequency": "3-4 meals per day with portion control",
                "hydration": "3-4 liters of water daily"
            },
            "Obese": {
                "exercise_focus": "low-impact cardio and basic strength",
                "exercise_intensity": "low to moderate, gradually increasing",
                "exercise_frequency": "start with 3-4 days, build to 5-6 days",
                "cardio_recommendation": "walking, swimming, or cycling 5-6 days, 30-45 minutes",
                "nutrition_focus": "moderate caloric deficit, whole foods, reduced processed foods",
                "meal_frequency": "3 structured meals with planned snacks",
                "hydration": "3-4 liters of water daily"
            }
        }
        
        # Define body fat percentage specific training recommendations
        self.body_fat_training = {
            "low": {  # < 10% for men, < 18% for women
                "training_style": "muscle building focus",
                "rep_range": "8-12 reps",
                "rest_periods": "60-90 seconds",
                "cardio": "minimal, 1-2 sessions per week",
                "nutrition": "caloric surplus, high protein"
            },
            "athletic": {  # 10-15% for men, 18-22% for women
                "training_style": "performance-based training",
                "rep_range": "varied (5-15 reps)",
                "rest_periods": "varies by goal",
                "cardio": "sport-specific conditioning",
                "nutrition": "maintenance or slight surplus, timed nutrition"
            },
            "fitness": {  # 15-20% for men, 22-28% for women
                "training_style": "balanced approach",
                "rep_range": "10-15 reps",
                "rest_periods": "45-60 seconds",
                "cardio": "2-3 sessions per week, mixed intensity",
                "nutrition": "slight deficit or maintenance"
            },
            "average": {  # 20-25% for men, 28-32% for women
                "training_style": "fat loss with muscle preservation",
                "rep_range": "12-15 reps",
                "rest_periods": "30-45 seconds",
                "cardio": "3-4 sessions per week, include HIIT",
                "nutrition": "moderate deficit, high protein"
            },
            "high": {  # >25% for men, >32% for women
                "training_style": "fat loss focus",
                "rep_range": "15-20 reps",
                "rest_periods": "30 seconds or less",
                "cardio": "4-5 sessions, mix of HIIT and steady state",
                "nutrition": "significant deficit, high protein, high fiber"
            }
        }
        
        # Define posture recommendations
        self.posture_recommendations = {
            "Excellent": {
                "focus": "maintenance",
                "exercises": ["general mobility work", "yoga", "pilates"],
                "frequency": "1-2 times per week"
            },
            "Good": {
                "focus": "minor improvements",
                "exercises": ["core strengthening", "shoulder mobility", "yoga"],
                "frequency": "2-3 times per week"
            },
            "Fair": {
                "focus": "targeted corrections",
                "exercises": ["core stabilization", "back strengthening", "chest stretches", "hip flexor stretches"],
                "frequency": "3-4 times per week"
            },
            "Poor": {
                "focus": "significant correction",
                "exercises": ["postural alignment exercises", "thoracic mobility", "core stabilization", "neck alignment"],
                "frequency": "daily corrective exercises"
            }
        }
        
        # Define symmetry recommendations
        self.symmetry_recommendations = {
            "high": {  # 80-100 score
                "focus": "maintenance",
                "exercises": ["bilateral movements", "compound exercises"],
                "notes": "Continue with balanced training approach"
            },
            "medium": {  # 60-80 score
                "focus": "minor corrections",
                "exercises": ["unilateral exercises", "core stabilization"],
                "notes": "Pay attention to form and balance during exercises"
            },
            "low": {  # <60 score
                "focus": "significant correction",
                "exercises": ["targeted unilateral work", "corrective exercises", "balance training"],
                "notes": "Consider consulting with a physical therapist for personalized corrections"
            }
        }
    
    def analyze_body_status(self, body_analysis_result):
        """
        Analyze body status based on MediaPipe pose estimation results
        """
        if not body_analysis_result["success"]:
            return {
                "success": False,
                "message": body_analysis_result.get("message", "Body analysis failed")
            }
        
        measurements = body_analysis_result["body_measurements"]
        
        # Extract body type and BMI category
        body_type = measurements["body_type"]
        bmi_category = measurements["bmi_category"]
        body_fat_percentage = measurements["body_fat_percentage"]
        posture_quality = measurements["posture_quality"]
        symmetry_score = measurements["symmetry_score"]
        
        # Determine body fat category
        body_fat_category = self._determine_body_fat_category(body_fat_percentage, "male")  # Default to male, should be from user data
        
        # Determine symmetry category
        symmetry_category = self._determine_symmetry_category(symmetry_score)
        
        # Get recommendations based on body type
        body_type_key = body_type.split(" ")[0]  # Get first word (e.g., "Inverted" from "Inverted Triangle")
        body_type_recommendations = self.body_type_exercises.get(body_type_key, self.body_type_exercises["Rectangle"])
        
        # Get recommendations based on BMI category
        bmi_recommendations = self.bmi_category_recommendations.get(bmi_category, self.bmi_category_recommendations["Normal weight"])
        
        # Get training recommendations based on body fat
        training_recommendations = self.body_fat_training.get(body_fat_category, self.body_fat_training["average"])
        
        # Get posture recommendations
        posture_recommendations = self.posture_recommendations.get(posture_quality, self.posture_recommendations["Fair"])
        
        # Get symmetry recommendations
        symmetry_recommendations = self.symmetry_recommendations.get(symmetry_category, self.symmetry_recommendations["medium"])
        
        return {
            "success": True,
            "body_measurements": measurements,
            "body_type_recommendations": body_type_recommendations,
            "bmi_recommendations": bmi_recommendations,
            "training_recommendations": training_recommendations,
            "body_fat_category": body_fat_category,
            "posture_recommendations": posture_recommendations,
            "symmetry_recommendations": symmetry_recommendations,
            "symmetry_category": symmetry_category
        }
    
    def _determine_body_fat_category(self, body_fat_percentage, gender):
        """
        Determine body fat category based on percentage and gender
        """
        if gender.lower() == "male":
            if body_fat_percentage < 10:
                return "low"
            elif 10 <= body_fat_percentage < 15:
                return "athletic"
            elif 15 <= body_fat_percentage < 20:
                return "fitness"
            elif 20 <= body_fat_percentage < 25:
                return "average"
            else:
                return "high"
        else:  # female
            if body_fat_percentage < 18:
                return "low"
            elif 18 <= body_fat_percentage < 22:
                return "athletic"
            elif 22 <= body_fat_percentage < 28:
                return "fitness"
            elif 28 <= body_fat_percentage < 32:
                return "average"
            else:
                return "high"
    
    def _determine_symmetry_category(self, symmetry_score):
        """
        Determine symmetry category based on symmetry score
        """
        if symmetry_score >= 80:
            return "high"
        elif symmetry_score >= 60:
            return "medium"
        else:
            return "low"
    
    def get_recommended_activities(self, user_data, body_analysis=None):
        """
        Based on user data and body analysis, recommend specific activities
        """
        recommended_activities = []
        
        # Extract user's fitness goal
        fitness_goal = user_data.get('fitness_goal', 'general fitness')
        
        # If we have body analysis, use it to prioritize activities
        body_type_focus = []
        if body_analysis and body_analysis.get("success", False):
            body_type_recommendations = body_analysis.get("body_type_recommendations", {})
            body_type_focus = body_type_recommendations.get("focus_areas", [])
            
            # Get recommended exercises from body analysis
            recommended_exercises = body_type_recommendations.get("recommended_exercises", [])
            posture_recommendations = body_analysis.get("posture_recommendations", {})
            posture_exercises = posture_recommendations.get("exercises", [])
            symmetry_recommendations = body_analysis.get("symmetry_recommendations", {})
            symmetry_exercises = symmetry_recommendations.get("exercises", [])
            
            # Add body type recommended exercises
            for exercise in recommended_exercises[:5]:
                priority = "high" if fitness_goal == "muscle_gain" else "medium"
                recommended_activities.append({
                    'equipment': 'bodyweight',
                    'activity': exercise,
                    'priority': priority,
                    'category': 'body type'
                })
            
            # Add posture exercises
            for exercise in posture_exercises:
                recommended_activities.append({
                    'equipment': 'bodyweight',
                    'activity': exercise,
                    'priority': "high" if posture_recommendations.get("focus") == "significant correction" else "medium",
                    'category': 'posture'
                })
            
            # Add symmetry exercises
            for exercise in symmetry_exercises:
                recommended_activities.append({
                    'equipment': 'bodyweight',
                    'activity': exercise,
                    'priority': "high" if symmetry_recommendations.get("focus") == "significant correction" else "medium",
                    'category': 'symmetry'
                })
        
        # If we don't have enough activities yet, add general exercises based on fitness goal
        if len(recommended_activities) < 8:
            if fitness_goal == 'weight_loss':
                general_exercises = [
                    "HIIT training", "circuit training", "cardio intervals", 
                    "jumping jacks", "mountain climbers", "burpees",
                    "jump rope", "cycling", "running"
                ]
            elif fitness_goal == 'muscle_gain':
                general_exercises = [
                    "push-ups", "pull-ups", "squats", "lunges", 
                    "deadlifts", "bench press", "shoulder press",
                    "rows", "planks", "dips"
                ]
            else:  # general fitness
                general_exercises = [
                    "yoga", "pilates", "bodyweight exercises", 
                    "swimming", "cycling", "jogging",
                    "functional training", "mobility work", "stretching"
                ]
            
            # Add general exercises
            for exercise in general_exercises[:8 - len(recommended_activities)]:
                recommended_activities.append({
                    'equipment': 'bodyweight',
                    'activity': exercise,
                    'priority': 'medium',
                    'category': 'general'
                })
        
        # Sort by priority
        recommended_activities.sort(key=lambda x: 0 if x['priority'] == 'high' else 1)
        
        return recommended_activities 