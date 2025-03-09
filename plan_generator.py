import random

class PlanGenerator:
    def __init__(self):
        # Exercise templates based on fitness goals
        self.exercise_templates = {
            'weight_loss': {
                'sessions_per_week': 4,
                'cardio_minutes': 30,
                'strength_exercises': 3,
                'sets': 3,
                'reps': '12-15',
                'rest': '30-45 seconds'
            },
            'muscle_gain': {
                'sessions_per_week': 4,
                'cardio_minutes': 15,
                'strength_exercises': 5,
                'sets': 4,
                'reps': '8-12',
                'rest': '60-90 seconds'
            },
            'general_fitness': {
                'sessions_per_week': 3,
                'cardio_minutes': 20,
                'strength_exercises': 4,
                'sets': 3,
                'reps': '10-12',
                'rest': '45-60 seconds'
            }
        }
        
        # Body type specific exercise templates
        self.body_type_templates = {
            'Inverted Triangle': {
                'sessions_per_week': 4,
                'cardio_minutes': 25,
                'strength_focus': 'lower_body',
                'strength_exercises': 4,
                'sets': 3,
                'reps': '10-15',
                'rest': '45-60 seconds'
            },
            'Pear': {
                'sessions_per_week': 4,
                'cardio_minutes': 25,
                'strength_focus': 'upper_body',
                'strength_exercises': 4,
                'sets': 3,
                'reps': '10-15',
                'rest': '45-60 seconds'
            },
            'Athletic': {
                'sessions_per_week': 5,
                'cardio_minutes': 20,
                'strength_focus': 'full_body',
                'strength_exercises': 5,
                'sets': 4,
                'reps': '8-12',
                'rest': '60-90 seconds'
            },
            'Rectangle': {
                'sessions_per_week': 4,
                'cardio_minutes': 20,
                'strength_focus': 'balanced',
                'strength_exercises': 4,
                'sets': 3,
                'reps': '10-15',
                'rest': '45-60 seconds'
            }
        }
        
        # Diet templates based on fitness goals
        self.diet_templates = {
            'weight_loss': {
                'calories_adjustment': -500,  # Caloric deficit
                'protein_ratio': 0.3,  # 30% of calories from protein
                'carb_ratio': 0.4,     # 40% of calories from carbs
                'fat_ratio': 0.3,      # 30% of calories from fat
                'meals_per_day': 4,
                'snacks': 1
            },
            'muscle_gain': {
                'calories_adjustment': 300,   # Caloric surplus
                'protein_ratio': 0.35,  # 35% of calories from protein
                'carb_ratio': 0.45,     # 45% of calories from carbs
                'fat_ratio': 0.2,       # 20% of calories from fat
                'meals_per_day': 5,
                'snacks': 2
            },
            'general_fitness': {
                'calories_adjustment': 0,     # Maintenance calories
                'protein_ratio': 0.25,  # 25% of calories from protein
                'carb_ratio': 0.5,      # 50% of calories from carbs
                'fat_ratio': 0.25,      # 25% of calories from fat
                'meals_per_day': 3,
                'snacks': 2
            }
        }
        
        # BMI category specific diet templates
        self.bmi_diet_templates = {
            'Underweight': {
                'calories_adjustment': 500,  # Caloric surplus
                'protein_ratio': 0.3,  # 30% of calories from protein
                'carb_ratio': 0.5,     # 50% of calories from carbs
                'fat_ratio': 0.2,      # 20% of calories from fat
                'meals_per_day': 5,
                'snacks': 2
            },
            'Normal weight': {
                'calories_adjustment': 0,  # Maintenance
                'protein_ratio': 0.25,  # 25% of calories from protein
                'carb_ratio': 0.5,     # 50% of calories from carbs
                'fat_ratio': 0.25,     # 25% of calories from fat
                'meals_per_day': 3,
                'snacks': 2
            },
            'Overweight': {
                'calories_adjustment': -300,  # Moderate deficit
                'protein_ratio': 0.3,  # 30% of calories from protein
                'carb_ratio': 0.4,     # 40% of calories from carbs
                'fat_ratio': 0.3,      # 30% of calories from fat
                'meals_per_day': 4,
                'snacks': 1
            },
            'Obese': {
                'calories_adjustment': -500,  # Larger deficit
                'protein_ratio': 0.35,  # 35% of calories from protein
                'carb_ratio': 0.35,     # 35% of calories from carbs
                'fat_ratio': 0.3,      # 30% of calories from fat
                'meals_per_day': 4,
                'snacks': 0
            }
        }
        
        # Food suggestions by category
        self.food_suggestions = {
            'protein': [
                'Chicken breast', 'Turkey', 'Lean beef', 'Fish (salmon, tuna, tilapia)', 
                'Eggs', 'Greek yogurt', 'Cottage cheese', 'Tofu', 'Tempeh', 
                'Lentils', 'Chickpeas', 'Protein powder'
            ],
            'carbs': [
                'Brown rice', 'Quinoa', 'Sweet potatoes', 'Oats', 'Whole grain bread',
                'Whole grain pasta', 'Fruits', 'Beans', 'Barley', 'Buckwheat'
            ],
            'fats': [
                'Avocado', 'Nuts (almonds, walnuts)', 'Seeds (chia, flax)', 'Olive oil',
                'Coconut oil', 'Nut butters', 'Fatty fish', 'Eggs', 'Dark chocolate'
            ],
            'vegetables': [
                'Broccoli', 'Spinach', 'Kale', 'Bell peppers', 'Carrots', 'Cauliflower',
                'Zucchini', 'Tomatoes', 'Cucumber', 'Asparagus', 'Brussels sprouts'
            ]
        }
        
        # Exercise categories
        self.exercise_categories = {
            'cardio': [
                'Running', 'Cycling', 'Swimming', 'Rowing', 'Jumping rope',
                'Elliptical training', 'Stair climbing', 'HIIT', 'Dancing'
            ],
            'upper_body': [
                'Push-ups', 'Pull-ups', 'Bench press', 'Shoulder press', 'Bicep curls',
                'Tricep dips', 'Rows', 'Lat pulldowns', 'Chest flies'
            ],
            'lower_body': [
                'Squats', 'Lunges', 'Deadlifts', 'Leg press', 'Calf raises',
                'Glute bridges', 'Leg curls', 'Leg extensions', 'Step-ups'
            ],
            'core': [
                'Planks', 'Crunches', 'Russian twists', 'Leg raises', 'Mountain climbers',
                'Bicycle crunches', 'Dead bugs', 'Ab rollouts', 'Side planks'
            ],
            'flexibility': [
                'Yoga', 'Pilates', 'Static stretching', 'Dynamic stretching',
                'Foam rolling', 'Mobility drills'
            ]
        }
        
        # Nutrient-rich foods for specific body needs
        self.nutrient_foods = {
            'muscle_building': [
                'Lean meats', 'Eggs', 'Greek yogurt', 'Cottage cheese', 'Whey protein',
                'Salmon', 'Tuna', 'Quinoa', 'Lentils', 'Chickpeas', 'Tofu'
            ],
            'fat_loss': [
                'Leafy greens', 'Berries', 'Lean protein', 'Green tea', 'Apple cider vinegar',
                'Chili peppers', 'Chia seeds', 'Greek yogurt', 'Eggs', 'Cruciferous vegetables'
            ],
            'energy_boosting': [
                'Bananas', 'Oats', 'Sweet potatoes', 'Honey', 'Coffee', 'Green tea',
                'Quinoa', 'Eggs', 'Apples', 'Oranges', 'Dark chocolate'
            ],
            'recovery': [
                'Tart cherries', 'Watermelon', 'Salmon', 'Turmeric', 'Ginger',
                'Spinach', 'Kale', 'Blueberries', 'Pineapple', 'Nuts', 'Seeds'
            ]
        }
    
    def calculate_bmr(self, user_data):
        """Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation"""
        try:
            weight = float(user_data.get('weight', 70))  # kg
            height = float(user_data.get('height', 170))  # cm
            age = int(user_data.get('age', 30))
            gender = user_data.get('gender', 'male').lower()
            
            if gender == 'male':
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age - 161
                
            return bmr
        except (ValueError, TypeError):
            # Default BMR if calculation fails
            return 1800
    
    def calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure based on activity level"""
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9
        }
        
        multiplier = activity_multipliers.get(activity_level, 1.375)
        return bmr * multiplier
    
    def generate_exercise_plan(self, user_data, body_analysis=None):
        """Generate a personalized exercise plan based on user data and body analysis"""
        # Map user's fitness goal to template
        fitness_goal = user_data.get('fitness_goal', 'general_fitness')
        
        # Default template based on fitness goal
        if fitness_goal == 'weight_loss':
            template = self.exercise_templates['weight_loss']
        elif fitness_goal == 'muscle_gain':
            template = self.exercise_templates['muscle_gain']
        else:
            template = self.exercise_templates['general_fitness']
        
        # If body analysis is available, adjust template based on body type
        if body_analysis and body_analysis.get('success', False):
            body_type = body_analysis['body_measurements']['body_type'].split(' ')[0]  # Get first word (e.g., "Inverted" from "Inverted Triangle")
            if body_type in self.body_type_templates:
                # Merge templates, prioritizing body type template
                body_template = self.body_type_templates[body_type]
                template = {**template, **body_template}
        
        # Create weekly plan structure
        weekly_plan = []
        
        # Generate workout days
        for day in range(1, template['sessions_per_week'] + 1):
            # Determine workout focus
            if 'strength_focus' in template and template['strength_focus'] != 'balanced':
                focus = template['strength_focus']
            else:
                focus = self._get_workout_focus(day, template['sessions_per_week'])
            
            workout = {
                'day': f'Day {day}',
                'focus': focus,
                'cardio': {
                    'duration': template['cardio_minutes'],
                    'exercises': self._select_cardio_exercises(body_analysis)
                },
                'strength': {
                    'exercises': self._select_strength_exercises(
                        template['strength_exercises'],
                        focus,
                        body_analysis
                    ),
                    'sets': template['sets'],
                    'reps': template['reps'],
                    'rest': template['rest']
                }
            }
            weekly_plan.append(workout)
        
        # Generate notes based on body analysis if available
        if body_analysis and body_analysis.get('success', False):
            notes = self._generate_exercise_notes_with_body_analysis(fitness_goal, body_analysis)
        else:
            notes = self._generate_exercise_notes(fitness_goal)
        
        return {
            'weekly_plan': weekly_plan,
            'sessions_per_week': template['sessions_per_week'],
            'notes': notes
        }
    
    def _get_workout_focus(self, day, total_days):
        """Determine the focus of the workout based on the day"""
        if total_days <= 3:
            if day == 1:
                return 'upper_body'
            elif day == 2:
                return 'lower_body'
            else:
                return 'full_body'
        else:
            if day % 4 == 1:
                return 'upper_body'
            elif day % 4 == 2:
                return 'lower_body'
            elif day % 4 == 3:
                return 'core'
            else:
                return 'full_body'
    
    def _select_cardio_exercises(self, body_analysis=None):
        """Select appropriate cardio exercises based on body analysis"""
        cardio_options = []
        
        # Always include bodyweight cardio options
        cardio_options.extend(['Jumping jacks', 'High knees', 'Burpees', 'Mountain climbers', 
                              'Running', 'Cycling', 'Jump rope', 'Swimming', 'Elliptical training'])
        
        # If body analysis is available, adjust cardio based on BMI category
        if body_analysis and body_analysis.get('success', False):
            bmi_category = body_analysis['body_measurements']['bmi_category']
            
            if bmi_category == 'Obese':
                # Filter for low-impact options
                low_impact = ['Walking', 'Cycling', 'Swimming', 'Elliptical training']
                cardio_options = [opt for opt in cardio_options if any(li in opt for li in low_impact)]
                # Add more low-impact options if list is too short
                if len(cardio_options) < 2:
                    cardio_options.extend(['Walking', 'Elliptical training'])
            
            elif bmi_category == 'Underweight':
                # Filter out high-intensity options
                high_intensity = ['Interval', 'HIIT', 'Burpees', 'Sprint']
                cardio_options = [opt for opt in cardio_options if not any(hi in opt for hi in high_intensity)]
            
            # Consider posture quality
            posture_quality = body_analysis['body_measurements'].get('posture_quality', 'Fair')
            if posture_quality in ['Poor', 'Fair']:
                # Add posture-friendly cardio options
                cardio_options.extend(['Swimming', 'Walking', 'Elliptical training'])
                # Remove high-impact options that might worsen posture issues
                cardio_options = [opt for opt in cardio_options if opt not in ['Burpees', 'Jump rope', 'High knees']]
        
        # Select 1-2 cardio exercises
        num_exercises = min(2, len(cardio_options))
        return random.sample(cardio_options, num_exercises)
    
    def _select_strength_exercises(self, num_exercises, focus, body_analysis=None):
        """Select appropriate strength exercises based on focus and body analysis"""
        strength_options = []
        
        # If body analysis is available, get recommended exercises
        recommended_exercises = []
        avoid_exercises = []
        
        if body_analysis and body_analysis.get('success', False):
            body_type_recommendations = body_analysis.get('body_type_recommendations', {})
            recommended_exercises = body_type_recommendations.get('recommended_exercises', [])
            avoid_exercises = body_type_recommendations.get('avoid_exercises', [])
            
            # Add posture-specific exercises if needed
            posture_recommendations = body_analysis.get('posture_recommendations', {})
            if posture_recommendations:
                posture_exercises = posture_recommendations.get('exercises', [])
                recommended_exercises.extend(posture_exercises)
            
            # Add symmetry-specific exercises if needed
            symmetry_recommendations = body_analysis.get('symmetry_recommendations', {})
            if symmetry_recommendations:
                symmetry_exercises = symmetry_recommendations.get('exercises', [])
                recommended_exercises.extend(symmetry_exercises)
        
        # Add bodyweight exercises based on focus
        if focus in ['upper_body', 'full_body']:
            strength_options.extend(['Push-ups', 'Dips', 'Inverted rows', 'Pike push-ups', 'Pull-ups'])
        if focus in ['lower_body', 'full_body']:
            strength_options.extend(['Bodyweight squats', 'Lunges', 'Glute bridges', 'Step-ups', 'Bulgarian split squats'])
        if focus in ['core', 'full_body']:
            strength_options.extend(['Planks', 'Crunches', 'Mountain climbers', 'Russian twists', 'Leg raises'])
        
        # Add dumbbell exercises
        if focus in ['upper_body', 'full_body']:
            strength_options.extend(['Dumbbell chest press', 'Dumbbell rows', 'Dumbbell shoulder press', 'Bicep curls', 'Tricep extensions'])
        if focus in ['lower_body', 'full_body']:
            strength_options.extend(['Dumbbell squats', 'Dumbbell lunges', 'Dumbbell deadlifts', 'Dumbbell step-ups', 'Dumbbell calf raises'])
        if focus in ['core', 'full_body']:
            strength_options.extend(['Dumbbell russian twists', 'Dumbbell side bends', 'Weighted crunches'])
        
        # Filter out exercises to avoid based on body analysis
        if avoid_exercises:
            strength_options = [ex for ex in strength_options if not any(avoid in ex.lower() for avoid in avoid_exercises)]
        
        # Prioritize recommended exercises based on body analysis
        prioritized_options = []
        if recommended_exercises:
            for option in strength_options:
                if any(rec in option.lower() for rec in recommended_exercises):
                    prioritized_options.append(option)
        
        # If we have enough prioritized options, use those
        if len(prioritized_options) >= num_exercises:
            return random.sample(prioritized_options, num_exercises)
        
        # Otherwise, combine prioritized options with other options
        remaining_slots = num_exercises - len(prioritized_options)
        remaining_options = [opt for opt in strength_options if opt not in prioritized_options]
        
        if remaining_options and remaining_slots > 0:
            return prioritized_options + random.sample(remaining_options, min(remaining_slots, len(remaining_options)))
        else:
            return prioritized_options
    
    def _generate_exercise_notes(self, fitness_goal):
        """Generate notes and tips based on fitness goal"""
        if fitness_goal == 'weight_loss':
            return [
                "Focus on keeping your heart rate elevated during workouts",
                "Try to minimize rest between exercises",
                "Consider adding an extra 10-15 minutes of cardio on rest days",
                "Track your workouts to ensure progressive overload"
            ]
        elif fitness_goal == 'muscle_gain':
            return [
                "Focus on proper form and controlled movements",
                "Gradually increase weights as you get stronger",
                "Ensure adequate protein intake and recovery between workouts",
                "Get 7-9 hours of quality sleep for optimal muscle recovery"
            ]
        else:
            return [
                "Mix up your routine regularly to prevent plateaus",
                "Listen to your body and adjust intensity as needed",
                "Stay consistent with your workout schedule",
                "Include both strength and cardio for balanced fitness"
            ]
    
    def _generate_exercise_notes_with_body_analysis(self, fitness_goal, body_analysis):
        """Generate notes and tips based on fitness goal and body analysis"""
        # Start with standard notes
        notes = self._generate_exercise_notes(fitness_goal)
        
        # Add body type specific notes
        body_type = body_analysis['body_measurements']['body_type'].split(' ')[0]
        bmi_category = body_analysis['body_measurements']['bmi_category']
        body_fat_category = body_analysis.get('body_fat_category', 'average')
        
        # Body type specific notes
        if body_type == 'Inverted':
            notes.append("Focus on lower body exercises to balance your physique")
            notes.append("Include hip and glute exercises to create proportion")
        elif body_type == 'Pear':
            notes.append("Emphasize upper body training to create balance")
            notes.append("Include shoulder and back exercises to widen your upper body")
        elif body_type == 'Athletic':
            notes.append("Focus on maintaining your balanced physique")
            notes.append("Include functional training to enhance your natural athleticism")
        elif body_type == 'Rectangle':
            notes.append("Include exercises that create curves and definition")
            notes.append("Focus on shoulder and glute exercises to create an hourglass shape")
        
        # BMI category specific notes
        if bmi_category == 'Underweight':
            notes.append("Focus on strength training with adequate rest between sets")
            notes.append("Limit cardio to 2-3 short sessions per week")
            notes.append("Ensure you're eating enough calories to support muscle growth")
        elif bmi_category == 'Overweight' or bmi_category == 'Obese':
            notes.append("Start with low-impact exercises and gradually increase intensity")
            notes.append("Focus on form and technique rather than intensity initially")
            notes.append("Include regular cardio sessions for heart health and calorie burning")
        
        # Body fat category specific notes
        if body_fat_category == 'low':
            notes.append("Focus on muscle building with adequate caloric intake")
            notes.append("Limit high-intensity cardio to maintain energy balance")
        elif body_fat_category == 'high':
            notes.append("Include both strength and cardio for optimal fat loss")
            notes.append("Consider circuit training to maximize calorie burn")
        
        return notes
    
    def generate_diet_plan(self, user_data, body_analysis=None):
        """Generate a personalized diet plan based on user data and body analysis"""
        # Map user's fitness goal to template
        fitness_goal = user_data.get('fitness_goal', 'general_fitness')
        
        # Default template based on fitness goal
        if fitness_goal == 'weight_loss':
            template = self.diet_templates['weight_loss']
        elif fitness_goal == 'muscle_gain':
            template = self.diet_templates['muscle_gain']
        else:
            template = self.diet_templates['general_fitness']
        
        # If body analysis is available, adjust template based on BMI category
        if body_analysis and body_analysis.get('success', False):
            bmi_category = body_analysis['body_measurements']['bmi_category']
            if bmi_category in self.bmi_diet_templates:
                bmi_template = self.bmi_diet_templates[bmi_category]
                
                # For conflicting goals (e.g., user wants muscle gain but is overweight),
                # find a middle ground for calories
                if (fitness_goal == 'muscle_gain' and bmi_category in ['Overweight', 'Obese']) or \
                   (fitness_goal == 'weight_loss' and bmi_category == 'Underweight'):
                    # Average the calorie adjustments
                    template['calories_adjustment'] = (template['calories_adjustment'] + bmi_template['calories_adjustment']) // 2
                else:
                    # Otherwise use the BMI-based template
                    template = bmi_template
        
        # Calculate caloric needs
        bmr = self.calculate_bmr(user_data)
        activity_level = user_data.get('activity_level', 'moderately_active')
        tdee = self.calculate_tdee(bmr, activity_level)
        
        # Adjust calories based on goal
        daily_calories = tdee + template['calories_adjustment']
        
        # Calculate macronutrients
        protein_calories = daily_calories * template['protein_ratio']
        carb_calories = daily_calories * template['carb_ratio']
        fat_calories = daily_calories * template['fat_ratio']
        
        # Convert to grams
        protein_grams = protein_calories / 4  # 4 calories per gram of protein
        carb_grams = carb_calories / 4       # 4 calories per gram of carbs
        fat_grams = fat_calories / 9         # 9 calories per gram of fat
        
        # Generate meal plan with body analysis if available
        meal_plan = self._generate_meal_plan(
            template['meals_per_day'],
            template['snacks'],
            user_data.get('dietary_restrictions', []),
            body_analysis
        )
        
        # Generate notes with body analysis if available
        if body_analysis and body_analysis.get('success', False):
            notes = self._generate_diet_notes_with_body_analysis(
                fitness_goal, 
                user_data.get('dietary_restrictions', []),
                body_analysis
            )
        else:
            notes = self._generate_diet_notes(fitness_goal, user_data.get('dietary_restrictions', []))
        
        return {
            'daily_calories': int(daily_calories),
            'macros': {
                'protein': int(protein_grams),
                'carbs': int(carb_grams),
                'fat': int(fat_grams)
            },
            'meal_plan': meal_plan,
            'notes': notes
        }
    
    def _generate_meal_plan(self, meals_per_day, snacks, dietary_restrictions, body_analysis=None):
        """Generate a sample meal plan structure with body analysis considerations"""
        meal_plan = []
        
        # Filter food suggestions based on dietary restrictions
        filtered_foods = self._filter_foods_by_restrictions(dietary_restrictions)
        
        # If body analysis is available, prioritize certain nutrients
        priority_nutrients = []
        if body_analysis and body_analysis.get('success', False):
            bmi_category = body_analysis['body_measurements']['bmi_category']
            body_fat_category = body_analysis.get('body_fat_category', 'average')
            
            if bmi_category == 'Underweight' or body_fat_category == 'low':
                priority_nutrients.append('muscle_building')
                priority_nutrients.append('energy_boosting')
            elif bmi_category in ['Overweight', 'Obese'] or body_fat_category == 'high':
                priority_nutrients.append('fat_loss')
            
            # Always include recovery nutrients
            priority_nutrients.append('recovery')
        
        # Generate meals
        for i in range(1, meals_per_day + 1):
            if i == 1:
                meal_type = 'Breakfast'
            elif i == meals_per_day:
                meal_type = 'Dinner'
            else:
                meal_type = f'Meal {i}'
            
            # Select foods, prioritizing nutrient-rich options if available
            protein = self._select_prioritized_food(filtered_foods['protein'], priority_nutrients, 'protein')
            carbs = self._select_prioritized_food(filtered_foods['carbs'], priority_nutrients, 'carbs')
            vegetables = self._select_prioritized_food(filtered_foods['vegetables'], priority_nutrients, 'vegetables')
            fats = self._select_prioritized_food(filtered_foods['fats'], priority_nutrients, 'fats') if random.random() > 0.3 else None
            
            meal = {
                'name': meal_type,
                'protein': protein,
                'carbs': carbs,
                'vegetables': vegetables,
                'fats': fats
            }
            meal_plan.append(meal)
        
        # Generate snacks
        for i in range(1, snacks + 1):
            snack_options = []
            
            # Create snack options based on priority nutrients if available
            if priority_nutrients:
                for nutrient in priority_nutrients:
                    if nutrient == 'muscle_building':
                        snack_options.append(f"Protein shake with {random.choice(filtered_foods['carbs'])}")
                        snack_options.append(f"Greek yogurt with {random.choice(filtered_foods['carbs'])}")
                    elif nutrient == 'fat_loss':
                        snack_options.append(f"Vegetable sticks with hummus")
                        snack_options.append(f"Greek yogurt with berries")
                    elif nutrient == 'energy_boosting':
                        snack_options.append(f"Banana with nut butter")
                        snack_options.append(f"Trail mix with nuts and dried fruit")
                    elif nutrient == 'recovery':
                        snack_options.append(f"Tart cherry juice with protein")
                        snack_options.append(f"Smoothie with berries and protein")
            
            # Add standard options if we don't have enough
            if len(snack_options) < 2:
                snack_options.extend([
                    f"{random.choice(filtered_foods['protein'])} with {random.choice(filtered_foods['vegetables'])}",
                    f"{random.choice(filtered_foods['carbs'])} with {random.choice(filtered_foods['fats'])}",
                    f"Protein shake with {random.choice(filtered_foods['carbs'])}",
                    f"Greek yogurt with {random.choice(filtered_foods['carbs'])}"
                ])
            
            snack = {
                'name': f'Snack {i}',
                'options': random.sample(snack_options, min(2, len(snack_options)))
            }
            meal_plan.append(snack)
        
        return meal_plan
    
    def _select_prioritized_food(self, food_list, priority_nutrients, food_type):
        """Select a food item, prioritizing nutrient-rich options if available"""
        if not priority_nutrients:
            return random.choice(food_list)
        
        # Create a list of foods that match priority nutrients
        prioritized_foods = []
        for nutrient in priority_nutrients:
            if nutrient in self.nutrient_foods:
                for food in self.nutrient_foods[nutrient]:
                    if any(food.lower() in item.lower() for item in food_list):
                        prioritized_foods.append(next(item for item in food_list if food.lower() in item.lower()))
        
        # If we found prioritized foods, select from those
        if prioritized_foods:
            return random.choice(prioritized_foods)
        else:
            return random.choice(food_list)
    
    def _filter_foods_by_restrictions(self, dietary_restrictions):
        """Filter food suggestions based on dietary restrictions"""
        filtered_foods = {
            'protein': self.food_suggestions['protein'].copy(),
            'carbs': self.food_suggestions['carbs'].copy(),
            'fats': self.food_suggestions['fats'].copy(),
            'vegetables': self.food_suggestions['vegetables'].copy()
        }
        
        # Apply filters based on restrictions
        if 'vegetarian' in dietary_restrictions:
            filtered_foods['protein'] = [p for p in filtered_foods['protein'] 
                                        if p not in ['Chicken breast', 'Turkey', 'Lean beef', 'Fish (salmon, tuna, tilapia)']]
        
        if 'vegan' in dietary_restrictions:
            filtered_foods['protein'] = [p for p in filtered_foods['protein'] 
                                        if p not in ['Chicken breast', 'Turkey', 'Lean beef', 'Fish (salmon, tuna, tilapia)', 
                                                    'Eggs', 'Greek yogurt', 'Cottage cheese']]
            filtered_foods['fats'] = [f for f in filtered_foods['fats'] 
                                     if f not in ['Eggs', 'Fatty fish']]
        
        if 'gluten_free' in dietary_restrictions:
            filtered_foods['carbs'] = [c for c in filtered_foods['carbs'] 
                                      if c not in ['Whole grain bread', 'Whole grain pasta', 'Barley']]
        
        if 'dairy_free' in dietary_restrictions:
            filtered_foods['protein'] = [p for p in filtered_foods['protein'] 
                                        if p not in ['Greek yogurt', 'Cottage cheese']]
        
        # Ensure we have at least some options in each category
        for category, foods in filtered_foods.items():
            if not foods:
                filtered_foods[category] = self.food_suggestions[category]
        
        return filtered_foods
    
    def _generate_diet_notes(self, fitness_goal, dietary_restrictions):
        """Generate notes and tips for the diet plan"""
        notes = []
        
        # Goal-specific notes
        if fitness_goal == 'weight_loss':
            notes.extend([
                "Focus on high-volume, low-calorie foods to stay full",
                "Drink water before meals to help with portion control",
                "Consider intermittent fasting if it suits your lifestyle",
                "Track your food intake to ensure you maintain a caloric deficit"
            ])
        elif fitness_goal == 'muscle_gain':
            notes.extend([
                "Prioritize protein intake, especially after workouts",
                "Don't skip meals - consistency is key for muscle growth",
                "Consider a protein shake within 30 minutes after training",
                "Eat a balanced meal 1-2 hours before workouts"
            ])
        else:
            notes.extend([
                "Focus on whole, minimally processed foods",
                "Stay hydrated throughout the day",
                "Eat a rainbow of fruits and vegetables for micronutrients",
                "Listen to your body's hunger and fullness cues"
            ])
        
        # Restriction-specific notes
        if 'vegetarian' in dietary_restrictions:
            notes.append("Combine plant proteins to ensure you get all essential amino acids")
        if 'vegan' in dietary_restrictions:
            notes.append("Consider B12 supplementation and monitor iron and calcium intake")
        if 'gluten_free' in dietary_restrictions:
            notes.append("Look for certified gluten-free products to avoid cross-contamination")
        if 'dairy_free' in dietary_restrictions:
            notes.append("Ensure adequate calcium intake from non-dairy sources")
        
        return notes
    
    def _generate_diet_notes_with_body_analysis(self, fitness_goal, dietary_restrictions, body_analysis):
        """Generate notes and tips for the diet plan based on body analysis"""
        # Start with standard notes
        notes = self._generate_diet_notes(fitness_goal, dietary_restrictions)
        
        # Add body analysis specific notes
        bmi_category = body_analysis['body_measurements']['bmi_category']
        body_fat_category = body_analysis.get('body_fat_category', 'average')
        
        # BMI category specific notes
        if bmi_category == 'Underweight':
            notes.append("Focus on nutrient-dense, calorie-rich foods")
            notes.append("Eat more frequently throughout the day")
            notes.append("Include healthy fats like nuts, avocados, and olive oil")
            notes.append("Consider liquid calories like smoothies for easier consumption")
        elif bmi_category == 'Overweight':
            notes.append("Focus on portion control and mindful eating")
            notes.append("Prioritize protein and fiber to increase satiety")
            notes.append("Limit added sugars and processed foods")
            notes.append("Stay hydrated to help control hunger")
        elif bmi_category == 'Obese':
            notes.append("Work with a healthcare provider for personalized nutrition advice")
            notes.append("Focus on whole foods and avoid processed items")
            notes.append("Consider keeping a food journal to increase awareness")
            notes.append("Prioritize protein at each meal to maintain muscle mass")
        
        # Body fat category specific notes
        if body_fat_category == 'low':
            notes.append("Ensure adequate fat intake for hormone production")
            notes.append("Focus on nutrient density rather than calorie restriction")
        elif body_fat_category == 'high':
            notes.append("Focus on lean proteins and high-fiber foods")
            notes.append("Consider timing carbohydrates around workouts")
            notes.append("Include metabolism-boosting foods like chili peppers and green tea")
        
        return notes