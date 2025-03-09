import numpy as np

class NutritionAnalyzer:
    def __init__(self):
        # Base metabolic rates by body type (calories per kg of body weight)
        self.bmr_multipliers = {
            'athletic': 24.0,
            'normal': 22.0,
            'overweight': 20.0
        }
        
        # Activity level multipliers
        self.activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        
        # Macronutrient ratios by fitness goal (protein/carbs/fats)
        self.macro_ratios = {
            'weight_loss': {'protein': 0.40, 'carbs': 0.30, 'fats': 0.30},
            'muscle_gain': {'protein': 0.30, 'carbs': 0.50, 'fats': 0.20},
            'maintenance': {'protein': 0.30, 'carbs': 0.40, 'fats': 0.30},
            'general_fitness': {'protein': 0.25, 'carbs': 0.45, 'fats': 0.30}
        }
        
        # Micronutrient focus by body type
        self.micronutrient_focus = {
            'athletic': [
                {'nutrient': 'Magnesium', 'amount': '400-500mg', 'sources': ['almonds', 'spinach', 'black beans']},
                {'nutrient': 'Zinc', 'amount': '15-20mg', 'sources': ['lean meats', 'pumpkin seeds', 'oysters']},
                {'nutrient': 'B-Complex', 'amount': '100% DV', 'sources': ['whole grains', 'eggs', 'leafy greens']}
            ],
            'normal': [
                {'nutrient': 'Vitamin D', 'amount': '1000-2000 IU', 'sources': ['fatty fish', 'egg yolks', 'fortified foods']},
                {'nutrient': 'Calcium', 'amount': '1000mg', 'sources': ['dairy', 'fortified plant milk', 'tofu']},
                {'nutrient': 'Iron', 'amount': '18mg', 'sources': ['lean meat', 'lentils', 'spinach']}
            ],
            'overweight': [
                {'nutrient': 'Chromium', 'amount': '200-400mcg', 'sources': ['whole grains', 'broccoli', 'grape juice']},
                {'nutrient': 'Fiber', 'amount': '25-35g', 'sources': ['vegetables', 'fruits', 'legumes']},
                {'nutrient': 'Vitamin C', 'amount': '500mg', 'sources': ['citrus fruits', 'bell peppers', 'berries']}
            ]
        }

    def calculate_nutrition_needs(self, user_data, body_analysis):
        """Calculate detailed nutrition needs based on body analysis and user data."""
        bmi_category = body_analysis['body_measurements']['bmi_category']
        activity_level = user_data['activity_level']
        fitness_goal = user_data['fitness_goal']
        weight = user_data.get('weight', 70)  # Default to 70kg if not provided
        
        # Calculate base calories
        bmr = self._calculate_bmr(weight, bmi_category)
        tdee = self._calculate_tdee(bmr, activity_level)
        target_calories = self._adjust_calories_for_goal(tdee, fitness_goal)
        
        # Calculate macronutrient needs
        macros = self._calculate_macros(target_calories, fitness_goal)
        
        # Get micronutrient recommendations
        micros = self._get_micronutrient_recommendations(bmi_category)
        
        # Get meal timing recommendations
        meal_timing = self._get_meal_timing_recommendations(activity_level, fitness_goal)
        
        # Get supplement recommendations
        supplements = self._get_supplement_recommendations(bmi_category, fitness_goal)
        
        return {
            'daily_calories': target_calories,
            'macronutrients': macros,
            'micronutrients': micros,
            'meal_timing': meal_timing,
            'supplements': supplements,
            'hydration': self._calculate_hydration_needs(weight, activity_level)
        }

    def _calculate_bmr(self, weight, bmi_category):
        """Calculate Basal Metabolic Rate."""
        multiplier = self.bmr_multipliers.get(bmi_category, self.bmr_multipliers['normal'])
        return weight * multiplier

    def _calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure."""
        multiplier = self.activity_multipliers.get(activity_level, self.activity_multipliers['moderate'])
        return bmr * multiplier

    def _adjust_calories_for_goal(self, tdee, fitness_goal):
        """Adjust calories based on fitness goal."""
        adjustments = {
            'weight_loss': -500,  # Caloric deficit
            'muscle_gain': 300,   # Caloric surplus
            'maintenance': 0,
            'general_fitness': 0
        }
        adjustment = adjustments.get(fitness_goal, 0)
        return tdee + adjustment

    def _calculate_macros(self, calories, fitness_goal):
        """Calculate macronutrient needs in grams."""
        ratios = self.macro_ratios.get(fitness_goal, self.macro_ratios['general_fitness'])
        
        protein_calories = calories * ratios['protein']
        carb_calories = calories * ratios['carbs']
        fat_calories = calories * ratios['fats']
        
        return {
            'protein': round(protein_calories / 4),  # 4 calories per gram
            'carbohydrates': round(carb_calories / 4),
            'fats': round(fat_calories / 9)  # 9 calories per gram
        }

    def _get_micronutrient_recommendations(self, bmi_category):
        """Get micronutrient recommendations based on body type."""
        return self.micronutrient_focus.get(bmi_category, self.micronutrient_focus['normal'])

    def _get_meal_timing_recommendations(self, activity_level, fitness_goal):
        """Get meal timing recommendations."""
        base_meals = {
            'weight_loss': [
                {'timing': 'Breakfast', 'size': 'moderate', 'focus': 'protein and fiber'},
                {'timing': 'Lunch', 'size': 'largest', 'focus': 'protein and vegetables'},
                {'timing': 'Dinner', 'size': 'moderate', 'focus': 'protein and vegetables'},
                {'timing': 'Snacks', 'size': 'small', 'focus': 'protein and fiber'}
            ],
            'muscle_gain': [
                {'timing': 'Breakfast', 'size': 'large', 'focus': 'protein and carbs'},
                {'timing': 'Pre-workout', 'size': 'moderate', 'focus': 'carbs and protein'},
                {'timing': 'Post-workout', 'size': 'large', 'focus': 'protein and carbs'},
                {'timing': 'Dinner', 'size': 'moderate', 'focus': 'protein and vegetables'},
                {'timing': 'Before bed', 'size': 'small', 'focus': 'protein and healthy fats'}
            ]
        }
        
        return base_meals.get(fitness_goal, base_meals['weight_loss'])

    def _get_supplement_recommendations(self, bmi_category, fitness_goal):
        """Get supplement recommendations."""
        base_supplements = {
            'essential': ['multivitamin', 'vitamin D', 'omega-3'],
            'athletic': ['creatine', 'protein powder', 'BCAAs'],
            'weight_loss': ['protein powder', 'fiber supplement', 'green tea extract'],
            'muscle_gain': ['creatine', 'protein powder', 'BCAAs', 'beta-alanine']
        }
        
        recommended = base_supplements['essential'].copy()
        
        if bmi_category == 'athletic':
            recommended.extend(base_supplements['athletic'])
        
        if fitness_goal in base_supplements:
            recommended.extend([s for s in base_supplements[fitness_goal] if s not in recommended])
        
        return recommended

    def _calculate_hydration_needs(self, weight, activity_level):
        """Calculate daily hydration needs."""
        # Base hydration: 30-35ml per kg of body weight
        base_hydration = weight * 33  # ml
        
        # Activity adjustment
        activity_adjustments = {
            'sedentary': 1.0,
            'light': 1.2,
            'moderate': 1.4,
            'active': 1.6,
            'very_active': 1.8
        }
        
        adjustment = activity_adjustments.get(activity_level, 1.0)
        total_hydration = base_hydration * adjustment
        
        return {
            'daily_water_ml': round(total_hydration),
            'daily_water_oz': round(total_hydration / 29.5735),  # Convert to fluid ounces
            'recommendations': [
                'Drink 500ml upon waking',
                'Drink 500ml 30 minutes before each meal',
                'Drink 250ml every hour during activity',
                'Monitor urine color (should be light yellow)'
            ]
        } 