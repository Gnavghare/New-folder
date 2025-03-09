class WorkoutPlanner:
    def __init__(self):
        # Exercise difficulty levels
        self.difficulty_levels = ['beginner', 'intermediate', 'advanced']
        
        # Exercise categories
        self.exercise_categories = {
            'strength': {
                'beginner': [
                    {'name': 'Bodyweight Squats', 'sets': 3, 'reps': '12-15'},
                    {'name': 'Push-ups (Modified)', 'sets': 3, 'reps': '8-12'},
                    {'name': 'Dumbbell Rows', 'sets': 3, 'reps': '12-15'},
                    {'name': 'Glute Bridges', 'sets': 3, 'reps': '15-20'}
                ],
                'intermediate': [
                    {'name': 'Barbell Squats', 'sets': 4, 'reps': '8-12'},
                    {'name': 'Push-ups', 'sets': 4, 'reps': '12-15'},
                    {'name': 'Bent-over Rows', 'sets': 4, 'reps': '8-12'},
                    {'name': 'Romanian Deadlifts', 'sets': 4, 'reps': '8-12'}
                ],
                'advanced': [
                    {'name': 'Front Squats', 'sets': 5, 'reps': '6-8'},
                    {'name': 'Weighted Push-ups', 'sets': 5, 'reps': '8-12'},
                    {'name': 'Weighted Pull-ups', 'sets': 4, 'reps': '6-8'},
                    {'name': 'Barbell Deadlifts', 'sets': 5, 'reps': '5-8'}
                ]
            },
            'cardio': {
                'beginner': [
                    {'name': 'Walking', 'duration': '30 min', 'intensity': 'moderate'},
                    {'name': 'Stationary Bike', 'duration': '20 min', 'intensity': 'light'},
                    {'name': 'Swimming', 'duration': '20 min', 'intensity': 'light'}
                ],
                'intermediate': [
                    {'name': 'Jogging', 'duration': '30 min', 'intensity': 'moderate'},
                    {'name': 'HIIT Cycling', 'duration': '25 min', 'intensity': 'high'},
                    {'name': 'Swimming', 'duration': '30 min', 'intensity': 'moderate'}
                ],
                'advanced': [
                    {'name': 'Running', 'duration': '45 min', 'intensity': 'high'},
                    {'name': 'HIIT Training', 'duration': '30 min', 'intensity': 'very high'},
                    {'name': 'Swimming', 'duration': '45 min', 'intensity': 'high'}
                ]
            },
            'flexibility': {
                'beginner': [
                    {'name': 'Basic Stretching', 'duration': '15 min', 'focus': 'full body'},
                    {'name': 'Yoga (Basic)', 'duration': '20 min', 'focus': 'mobility'},
                    {'name': 'Dynamic Warm-up', 'duration': '10 min', 'focus': 'preparation'}
                ],
                'intermediate': [
                    {'name': 'Advanced Stretching', 'duration': '20 min', 'focus': 'full body'},
                    {'name': 'Yoga (Intermediate)', 'duration': '30 min', 'focus': 'strength and flexibility'},
                    {'name': 'Mobility Work', 'duration': '15 min', 'focus': 'joint mobility'}
                ],
                'advanced': [
                    {'name': 'Dynamic Flexibility', 'duration': '25 min', 'focus': 'full body'},
                    {'name': 'Yoga (Advanced)', 'duration': '45 min', 'focus': 'power and flexibility'},
                    {'name': 'Movement Flow', 'duration': '20 min', 'focus': 'coordination'}
                ]
            }
        }
        
        # Posture correction exercises
        self.posture_exercises = {
            'forward_head': [
                {'name': 'Chin Tucks', 'sets': 3, 'reps': '10-12'},
                {'name': 'Wall Angels', 'sets': 3, 'reps': '8-10'},
                {'name': 'Upper Trapezius Stretch', 'duration': '30 sec', 'sides': 'both'}
            ],
            'rounded_shoulders': [
                {'name': 'Band Pull-Aparts', 'sets': 3, 'reps': '15-20'},
                {'name': 'Face Pulls', 'sets': 3, 'reps': '12-15'},
                {'name': 'Doorway Stretch', 'duration': '30 sec', 'sides': 'both'}
            ],
            'anterior_pelvic_tilt': [
                {'name': 'Glute Bridges', 'sets': 3, 'reps': '12-15'},
                {'name': 'Dead Bugs', 'sets': 3, 'reps': '10 each side'},
                {'name': 'Hip Flexor Stretch', 'duration': '45 sec', 'sides': 'both'}
            ]
        }
        
        # Recovery protocols
        self.recovery_protocols = {
            'active': [
                {'name': 'Light Walking', 'duration': '20-30 min'},
                {'name': 'Mobility Work', 'duration': '15-20 min'},
                {'name': 'Foam Rolling', 'duration': '10-15 min'}
            ],
            'passive': [
                {'name': 'Static Stretching', 'duration': '15-20 min'},
                {'name': 'Meditation', 'duration': '10-15 min'},
                {'name': 'Deep Breathing', 'duration': '5-10 min'}
            ]
        }

    def create_workout_plan(self, user_data, body_analysis):
        """Create a personalized workout plan based on user data and body analysis."""
        fitness_level = self._determine_fitness_level(user_data, body_analysis)
        posture_issues = body_analysis['body_measurements']['posture_issues']
        
        # Get base workout structure
        workouts = self._get_base_workouts(fitness_level, user_data['fitness_goal'])
        
        # Add posture correction exercises
        posture_work = self._get_posture_exercises(posture_issues)
        
        # Get recovery protocols
        recovery = self._get_recovery_protocol(user_data['activity_level'])
        
        # Create weekly schedule
        schedule = self._create_weekly_schedule(
            workouts, 
            posture_work,
            recovery,
            user_data['activity_level']
        )
        
        return {
            'fitness_level': fitness_level,
            'weekly_schedule': schedule,
            'progression_plan': self._create_progression_plan(fitness_level),
            'recovery_protocol': recovery
        }

    def _determine_fitness_level(self, user_data, body_analysis):
        """Determine user's fitness level based on various factors."""
        points = 0
        
        # Activity level contribution
        activity_points = {
            'sedentary': 0,
            'light': 1,
            'moderate': 2,
            'active': 3,
            'very_active': 4
        }
        points += activity_points.get(user_data['activity_level'], 0)
        
        # Body composition contribution
        if body_analysis['body_measurements']['bmi_category'] == 'athletic':
            points += 2
        
        # Posture quality contribution
        if body_analysis['body_measurements']['posture_quality'] == 'good':
            points += 1
        
        # Determine level based on points
        if points <= 2:
            return 'beginner'
        elif points <= 4:
            return 'intermediate'
        else:
            return 'advanced'

    def _get_base_workouts(self, fitness_level, fitness_goal):
        """Get base workout exercises based on fitness level and goal."""
        workouts = {
            'strength': self.exercise_categories['strength'][fitness_level],
            'cardio': self.exercise_categories['cardio'][fitness_level],
            'flexibility': self.exercise_categories['flexibility'][fitness_level]
        }
        
        # Adjust volume based on goal
        if fitness_goal == 'muscle_gain':
            for exercise in workouts['strength']:
                exercise['sets'] = min(exercise['sets'] + 1, 6)
        elif fitness_goal == 'weight_loss':
            for exercise in workouts['cardio']:
                exercise['duration'] = str(int(exercise['duration'].split()[0]) + 10) + ' min'
        
        return workouts

    def _get_posture_exercises(self, posture_issues):
        """Get specific exercises for posture correction."""
        exercises = []
        for issue in posture_issues:
            if issue in self.posture_exercises:
                exercises.extend(self.posture_exercises[issue])
        return exercises if exercises else self.posture_exercises['forward_head']  # Default exercises

    def _get_recovery_protocol(self, activity_level):
        """Get appropriate recovery protocol based on activity level."""
        if activity_level in ['active', 'very_active']:
            return {
                'active_recovery': self.recovery_protocols['active'],
                'passive_recovery': self.recovery_protocols['passive']
            }
        return {'passive_recovery': self.recovery_protocols['passive']}

    def _create_weekly_schedule(self, workouts, posture_work, recovery, activity_level):
        """Create a weekly workout schedule."""
        # Determine number of training days
        training_days = {
            'sedentary': 3,
            'light': 3,
            'moderate': 4,
            'active': 5,
            'very_active': 6
        }
        days = training_days.get(activity_level, 3)
        
        schedule = {}
        
        # Create schedule based on number of training days
        if days <= 3:
            schedule = {
                'Monday': {
                    'main': workouts['strength'][:2] + workouts['cardio'][:1],
                    'posture': posture_work[:2],
                    'flexibility': workouts['flexibility'][:1]
                },
                'Wednesday': {
                    'main': workouts['strength'][2:] + workouts['cardio'][1:2],
                    'posture': posture_work[2:],
                    'flexibility': workouts['flexibility'][1:2]
                },
                'Friday': {
                    'main': workouts['strength'] + workouts['cardio'][2:],
                    'posture': posture_work,
                    'flexibility': workouts['flexibility'][2:]
                },
                'Tuesday': {'recovery': recovery['passive_recovery']},
                'Thursday': {'recovery': recovery['passive_recovery']},
                'Saturday': {'recovery': recovery.get('active_recovery', recovery['passive_recovery'])},
                'Sunday': {'rest': True}
            }
        else:
            # Create 5-day split for more advanced schedules
            schedule = {
                'Monday': {
                    'main': workouts['strength'][:2],
                    'posture': posture_work[:2],
                    'cardio': workouts['cardio'][:1]
                },
                'Tuesday': {
                    'main': workouts['strength'][2:],
                    'flexibility': workouts['flexibility'][:2]
                },
                'Wednesday': {
                    'main': workouts['cardio'][1:],
                    'posture': posture_work[2:],
                    'flexibility': workouts['flexibility'][2:]
                },
                'Thursday': {
                    'main': workouts['strength'],
                    'posture': posture_work[:2]
                },
                'Friday': {
                    'main': workouts['cardio'],
                    'flexibility': workouts['flexibility']
                },
                'Saturday': {'recovery': recovery.get('active_recovery', recovery['passive_recovery'])},
                'Sunday': {'rest': True}
            }
        
        return schedule

    def _create_progression_plan(self, fitness_level):
        """Create a progression plan based on fitness level."""
        progressions = {
            'beginner': {
                'weeks_1_4': 'Focus on form and building base strength',
                'weeks_5_8': 'Increase weights by 5-10% or add 1-2 reps',
                'weeks_9_12': 'Introduce more complex movements'
            },
            'intermediate': {
                'weeks_1_4': 'Increase intensity of main lifts',
                'weeks_5_8': 'Add advanced variations of exercises',
                'weeks_9_12': 'Implement progressive overload protocols'
            },
            'advanced': {
                'weeks_1_4': 'Focus on power and explosive movements',
                'weeks_5_8': 'Implement periodization strategies',
                'weeks_9_12': 'Peak strength and performance phase'
            }
        }
        return progressions[fitness_level] 