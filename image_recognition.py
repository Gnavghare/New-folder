import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import keras
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input # type: ignore
from keras.applications.resnet import ResNet50 # type: ignore
from PIL import Image
import tensorflow_hub as hub
import math
import logging

class ImageRecognitionEngine:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        
        # Initialize pose detection with high accuracy settings
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize holistic detection for comprehensive body analysis
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Load pre-trained models for various analyses
        self.body_composition_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        self.posture_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Initialize segmentation model
        self.segmentation_model = hub.load('https://tfhub.dev/tensorflow/resnet_50/classification/1')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_image(self, image_path, required_analyses=None):
        """
        Process image with comprehensive analysis options
        
        Args:
            image_path: Path to the image file
            required_analyses: List of required analyses (e.g., ['pose', 'body_composition', 'measurements'])
        """
        try:
            # Read and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                raise ValueError("Failed to load image")

            # Initialize results dictionary
            results = {
                'success': True,
                'analyses': {},
                'errors': []
            }

            # Perform requested analyses
            analyses_map = {
                'pose': self._analyze_pose,
                'body_composition': self._analyze_body_composition,
                'measurements': self._analyze_measurements,
                'posture': self._analyze_posture,
                'symmetry': self._analyze_symmetry,
                'muscle_groups': self._analyze_muscle_groups,
                'body_fat': self._analyze_body_fat,
                'proportions': self._analyze_body_proportions
            }

            # If no specific analyses requested, perform all
            if not required_analyses:
                required_analyses = list(analyses_map.keys())

            # Perform each requested analysis
            for analysis in required_analyses:
                try:
                    if analysis in analyses_map:
                        results['analyses'][analysis] = analyses_map[analysis](image)
                except Exception as e:
                    results['errors'].append(f"Error in {analysis}: {str(e)}")
                    results['analyses'][analysis] = None

            return results

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _load_and_preprocess_image(self, image_path):
        """Enhanced image loading and preprocessing"""
        try:
            # Read image using multiple methods
            image = None
            try:
                image = cv2.imread(image_path)
                if image is None:
                    image = np.array(Image.open(image_path))
            except:
                try:
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_image(image)
                except Exception as e:
                    raise ValueError(f"Failed to load image: {str(e)}")

            # Image validation
            if image is None or image.size == 0:
                raise ValueError("Invalid image data")

            # Convert to RGB if necessary
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Enhance image quality
            image = self._enhance_image_quality(image)
            
            return image

        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            return None

    def _enhance_image_quality(self, image):
        """Apply various image enhancement techniques"""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            # Merge channels
            enhanced_lab = cv2.merge((cl,a,b))
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # Denoise
            enhanced_image = cv2.fastNlMeansDenoisingColored(enhanced_image)

            # Sharpen
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

            return enhanced_image

        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}")
            return image

    def _analyze_pose(self, image):
        """Advanced pose analysis with improved landmark detection"""
        try:
            # Process with MediaPipe Pose
            results = self.pose.process(image)
            
            if not results.pose_landmarks:
                raise ValueError("No pose landmarks detected")

            # Extract landmarks with confidence scores
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            # Create visualization
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            return {
                'landmarks': landmarks,
                'pose_world_landmarks': results.pose_world_landmarks,
                'segmentation_mask': results.segmentation_mask,
                'annotated_image': annotated_image
            }

        except Exception as e:
            raise ValueError(f"Pose analysis failed: {str(e)}")

    def _analyze_body_composition(self, image):
        """Analyze body composition using advanced techniques"""
        try:
            # Prepare image for body composition analysis
            img_array = cv2.resize(image, (224, 224))
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Get features from MobileNetV2
            features = self.body_composition_model.predict(img_array)

            # Analyze body segments
            segmentation_mask = self._get_body_segmentation(image)
            
            # Calculate body composition metrics
            composition_metrics = self._calculate_composition_metrics(features, segmentation_mask)

            return {
                'body_type': composition_metrics['body_type'],
                'muscle_mass_index': composition_metrics['muscle_mass_index'],
                'body_fat_percentage': composition_metrics['body_fat_percentage'],
                'segmentation_mask': segmentation_mask
            }

        except Exception as e:
            raise ValueError(f"Body composition analysis failed: {str(e)}")

    def _analyze_measurements(self, image):
        """Extract detailed body measurements"""
        try:
            # Get pose landmarks
            pose_results = self.pose.process(image)
            if not pose_results.pose_landmarks:
                raise ValueError("No landmarks detected for measurements")

            landmarks = pose_results.pose_landmarks.landmark
            image_height, image_width = image.shape[:2]

            # Calculate key measurements
            measurements = {
                'height': self._calculate_height(landmarks, image_height),
                'shoulder_width': self._calculate_shoulder_width(landmarks, image_width),
                'chest_width': self._calculate_chest_width(landmarks, image_width),
                'waist_width': self._calculate_waist_width(landmarks, image_width),
                'hip_width': self._calculate_hip_width(landmarks, image_width),
                'inseam': self._calculate_inseam(landmarks, image_height),
                'arm_length': self._calculate_arm_length(landmarks, image_width),
                'leg_length': self._calculate_leg_length(landmarks, image_height)
            }

            # Calculate body ratios
            ratios = {
                'shoulder_to_waist': measurements['shoulder_width'] / measurements['waist_width'],
                'waist_to_hip': measurements['waist_width'] / measurements['hip_width'],
                'arm_to_height': measurements['arm_length'] / measurements['height'],
                'leg_to_height': measurements['leg_length'] / measurements['height']
            }

            return {
                'measurements': measurements,
                'ratios': ratios
            }

        except Exception as e:
            raise ValueError(f"Measurement analysis failed: {str(e)}")

    def _analyze_posture(self, image):
        """Advanced posture analysis"""
        try:
            # Get pose landmarks
            results = self.pose.process(image)
            if not results.pose_landmarks:
                raise ValueError("No landmarks detected for posture analysis")

            landmarks = results.pose_landmarks.landmark

            # Analyze various posture aspects
            posture_metrics = {
                'head_position': self._analyze_head_position(landmarks),
                'shoulder_alignment': self._analyze_shoulder_alignment(landmarks),
                'spine_alignment': self._analyze_spine_alignment(landmarks),
                'hip_alignment': self._analyze_hip_alignment(landmarks),
                'knee_alignment': self._analyze_knee_alignment(landmarks)
            }

            # Calculate overall posture score
            posture_score = self._calculate_posture_score(posture_metrics)

            # Generate recommendations
            recommendations = self._generate_posture_recommendations(posture_metrics)

            return {
                'metrics': posture_metrics,
                'overall_score': posture_score,
                'recommendations': recommendations
            }

        except Exception as e:
            raise ValueError(f"Posture analysis failed: {str(e)}")

    def _analyze_symmetry(self, image):
        """Analyze body symmetry"""
        try:
            results = self.pose.process(image)
            if not results.pose_landmarks:
                raise ValueError("No landmarks detected for symmetry analysis")

            landmarks = results.pose_landmarks.landmark

            # Analyze symmetry for different body parts
            symmetry_scores = {
                'shoulders': self._calculate_shoulder_symmetry(landmarks),
                'arms': self._calculate_arm_symmetry(landmarks),
                'hips': self._calculate_hip_symmetry(landmarks),
                'legs': self._calculate_leg_symmetry(landmarks)
            }

            # Calculate overall symmetry score
            overall_score = sum(symmetry_scores.values()) / len(symmetry_scores)

            # Generate recommendations based on asymmetries
            recommendations = self._generate_symmetry_recommendations(symmetry_scores)

            return {
                'scores': symmetry_scores,
                'overall_score': overall_score,
                'recommendations': recommendations
            }

        except Exception as e:
            raise ValueError(f"Symmetry analysis failed: {str(e)}")

    def _analyze_muscle_groups(self, image):
        """Analyze visible muscle groups and their development"""
        try:
            # Process with holistic detection
            results = self.holistic.process(image)
            if not results.pose_landmarks:
                raise ValueError("No landmarks detected for muscle analysis")

            # Analyze different muscle groups
            muscle_analysis = {
                'shoulders': self._analyze_shoulder_development(image, results),
                'arms': self._analyze_arm_development(image, results),
                'chest': self._analyze_chest_development(image, results),
                'back': self._analyze_back_development(image, results),
                'core': self._analyze_core_development(image, results),
                'legs': self._analyze_leg_development(image, results)
            }

            return {
                'muscle_groups': muscle_analysis,
                'recommendations': self._generate_muscle_recommendations(muscle_analysis)
            }

        except Exception as e:
            raise ValueError(f"Muscle group analysis failed: {str(e)}")

    def _analyze_body_fat(self, image):
        """Estimate body fat percentage using visual analysis"""
        try:
            # Get body segmentation
            segmentation_mask = self._get_body_segmentation(image)
            
            # Analyze body composition features
            features = self._extract_body_fat_features(image, segmentation_mask)
            
            # Estimate body fat percentage
            body_fat_percentage = self._estimate_body_fat_percentage(features)
            
            # Determine fat distribution pattern
            fat_distribution = self._analyze_fat_distribution(features)

            return {
                'body_fat_percentage': body_fat_percentage,
                'fat_distribution_pattern': fat_distribution,
                'health_risk_level': self._calculate_health_risk(body_fat_percentage, fat_distribution)
            }

        except Exception as e:
            raise ValueError(f"Body fat analysis failed: {str(e)}")

    def _analyze_body_proportions(self, image):
        """Analyze body proportions and aesthetic ratios"""
        try:
            results = self.pose.process(image)
            if not results.pose_landmarks:
                raise ValueError("No landmarks detected for proportion analysis")

            landmarks = results.pose_landmarks.landmark

            # Calculate various proportion ratios
            proportions = {
                'shoulder_to_waist': self._calculate_shoulder_to_waist_ratio(landmarks),
                'waist_to_hip': self._calculate_waist_to_hip_ratio(landmarks),
                'leg_to_torso': self._calculate_leg_to_torso_ratio(landmarks),
                'arm_to_torso': self._calculate_arm_to_torso_ratio(landmarks),
                'golden_ratio': self._calculate_golden_ratio_conformity(landmarks)
            }

            return {
                'proportions': proportions,
                'aesthetic_score': self._calculate_aesthetic_score(proportions),
                'recommendations': self._generate_proportion_recommendations(proportions)
            }

        except Exception as e:
            raise ValueError(f"Body proportion analysis failed: {str(e)}")

    # Helper methods for measurements
    def _calculate_height(self, landmarks, image_height):
        """Calculate height using landmarks"""
        top = min(landmarks[0].y, landmarks[1].y, landmarks[2].y)  # Head landmarks
        bottom = max(landmarks[27].y, landmarks[28].y)  # Ankle landmarks
        return abs(top - bottom) * image_height

    def _calculate_shoulder_width(self, landmarks, image_width):
        """Calculate shoulder width"""
        return abs(landmarks[11].x - landmarks[12].x) * image_width

    def _calculate_chest_width(self, landmarks, image_width):
        """Calculate chest width"""
        chest_left = landmarks[11].x
        chest_right = landmarks[12].x
        return abs(chest_left - chest_right) * image_width

    def _calculate_waist_width(self, landmarks, image_width):
        """Calculate waist width"""
        waist_left = (landmarks[11].x + landmarks[23].x) / 2
        waist_right = (landmarks[12].x + landmarks[24].x) / 2
        return abs(waist_left - waist_right) * image_width

    def _calculate_hip_width(self, landmarks, image_width):
        """Calculate hip width"""
        return abs(landmarks[23].x - landmarks[24].x) * image_width

    def _calculate_inseam(self, landmarks, image_height):
        """Calculate inseam length"""
        hip = (landmarks[23].y + landmarks[24].y) / 2
        ankle = (landmarks[27].y + landmarks[28].y) / 2
        return abs(hip - ankle) * image_height

    def _calculate_arm_length(self, landmarks, image_width):
        """Calculate arm length"""
        shoulder = landmarks[11]  # Left shoulder
        elbow = landmarks[13]    # Left elbow
        wrist = landmarks[15]    # Left wrist
        
        upper_arm = math.sqrt((shoulder.x - elbow.x)**2 + (shoulder.y - elbow.y)**2)
        forearm = math.sqrt((elbow.x - wrist.x)**2 + (elbow.y - wrist.y)**2)
        
        return (upper_arm + forearm) * image_width

    def _calculate_leg_length(self, landmarks, image_height):
        """Calculate leg length"""
        hip = landmarks[23]      # Left hip
        knee = landmarks[25]     # Left knee
        ankle = landmarks[27]    # Left ankle
        
        thigh = math.sqrt((hip.x - knee.x)**2 + (hip.y - knee.y)**2)
        shin = math.sqrt((knee.x - ankle.x)**2 + (knee.y - ankle.y)**2)
        
        return (thigh + shin) * image_height

    # Helper methods for posture analysis
    def _analyze_head_position(self, landmarks):
        """Analyze head position relative to shoulders"""
        nose = landmarks[0]
        mid_shoulder = ((landmarks[11].x + landmarks[12].x) / 2,
                       (landmarks[11].y + landmarks[12].y) / 2)
        
        forward_tilt = abs(nose.x - mid_shoulder[0])
        vertical_alignment = abs(nose.y - mid_shoulder[1])
        
        return {
            'forward_head': forward_tilt > 0.1,
            'tilt_angle': math.degrees(math.atan2(forward_tilt, vertical_alignment))
        }

    def _analyze_shoulder_alignment(self, landmarks):
        """Analyze shoulder alignment"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        height_diff = abs(left_shoulder.y - right_shoulder.y)
        rotation = abs(left_shoulder.z - right_shoulder.z)
        
        return {
            'height_difference': height_diff,
            'rotation': rotation,
            'is_aligned': height_diff < 0.05 and rotation < 0.05
        }

    def _analyze_spine_alignment(self, landmarks):
        """Analyze spine alignment"""
        # Calculate spine curve using multiple points
        spine_points = [
            landmarks[0],  # nose
            landmarks[11], # left shoulder
            landmarks[12], # right shoulder
            landmarks[23], # left hip
            landmarks[24]  # right hip
        ]
        
        # Calculate deviation from vertical
        x_coords = [point.x for point in spine_points]
        y_coords = [point.y for point in spine_points]
        
        # Fit a line to the spine points
        coeffs = np.polyfit(y_coords, x_coords, 1)
        deviation = abs(coeffs[0])  # Slope indicates deviation from vertical
        
        return {
            'lateral_deviation': deviation,
            'is_aligned': deviation < 0.05
        }

    # Helper methods for symmetry analysis
    def _calculate_shoulder_symmetry(self, landmarks):
        """Calculate shoulder symmetry score"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        mid_spine = landmarks[0]  # Using nose as reference
        
        left_distance = math.sqrt((left_shoulder.x - mid_spine.x)**2 + 
                                (left_shoulder.y - mid_spine.y)**2)
        right_distance = math.sqrt((right_shoulder.x - mid_spine.x)**2 + 
                                 (right_shoulder.y - mid_spine.y)**2)
        
        symmetry = 1 - abs(left_distance - right_distance)
        return max(0, min(1, symmetry))

    def _calculate_arm_symmetry(self, landmarks):
        """Calculate arm symmetry score"""
        # Left arm
        left_upper = math.sqrt((landmarks[11].x - landmarks[13].x)**2 + 
                             (landmarks[11].y - landmarks[13].y)**2)
        left_lower = math.sqrt((landmarks[13].x - landmarks[15].x)**2 + 
                             (landmarks[13].y - landmarks[15].y)**2)
        
        # Right arm
        right_upper = math.sqrt((landmarks[12].x - landmarks[14].x)**2 + 
                              (landmarks[12].y - landmarks[14].y)**2)
        right_lower = math.sqrt((landmarks[14].x - landmarks[16].x)**2 + 
                              (landmarks[14].y - landmarks[16].y)**2)
        
        upper_symmetry = 1 - abs(left_upper - right_upper)
        lower_symmetry = 1 - abs(left_lower - right_lower)
        
        return (upper_symmetry + lower_symmetry) / 2

    def _generate_symmetry_recommendations(self, symmetry_scores):
        """Generate exercise recommendations based on symmetry scores"""
        recommendations = []
        
        if symmetry_scores['shoulders'] < 0.8:
            recommendations.append({
                'area': 'shoulders',
                'exercises': ['unilateral shoulder press', 'face pulls', 'band pull-aparts'],
                'priority': 'high'
            })
            
        if symmetry_scores['arms'] < 0.8:
            recommendations.append({
                'area': 'arms',
                'exercises': ['single arm curls', 'unilateral tricep extensions'],
                'priority': 'medium'
            })
            
        if symmetry_scores['legs'] < 0.8:
            recommendations.append({
                'area': 'legs',
                'exercises': ['single leg press', 'bulgarian split squats'],
                'priority': 'high'
            })
            
        return recommendations

    def _generate_posture_recommendations(self, posture_metrics):
        """Generate exercise recommendations based on posture analysis"""
        recommendations = []
        
        if not posture_metrics['head_position']['is_aligned']:
            recommendations.append({
                'issue': 'forward_head',
                'exercises': ['chin tucks', 'wall slides', 'neck retraction'],
                'frequency': 'daily'
            })
            
        if not posture_metrics['shoulder_alignment']['is_aligned']:
            recommendations.append({
                'issue': 'shoulder_imbalance',
                'exercises': ['face pulls', 'band pull-aparts', 'wall angels'],
                'frequency': '3x per week'
            })
            
        if not posture_metrics['spine_alignment']['is_aligned']:
            recommendations.append({
                'issue': 'spine_alignment',
                'exercises': ['cat-cow stretch', 'bird dog', 'dead bug'],
                'frequency': 'daily'
            })
            
        return recommendations

    def _generate_proportion_recommendations(self, proportions):
        """Generate recommendations for improving body proportions"""
        recommendations = []
        
        # Shoulder to waist ratio
        if proportions['shoulder_to_waist'] < 1.4:
            recommendations.append({
                'focus': 'upper_body',
                'exercises': ['lateral raises', 'overhead press', 'pull-ups'],
                'priority': 'high'
            })
            
        # Waist to hip ratio
        if proportions['waist_to_hip'] > 0.9:
            recommendations.append({
                'focus': 'core',
                'exercises': ['planks', 'russian twists', 'vacuum exercises'],
                'priority': 'medium'
            })
            
        return recommendations

    def _calculate_aesthetic_score(self, proportions):
        """Calculate overall aesthetic score based on classical proportions"""
        weights = {
            'shoulder_to_waist': 0.3,
            'waist_to_hip': 0.2,
            'leg_to_torso': 0.2,
            'arm_to_torso': 0.15,
            'golden_ratio': 0.15
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in proportions:
                score += self._score_proportion(proportions[metric], metric) * weight
                
        return score

    def _score_proportion(self, value, metric_type):
        """Score individual proportion metrics"""
        ideal_values = {
            'shoulder_to_waist': 1.4,
            'waist_to_hip': 0.75,
            'leg_to_torso': 1.0,
            'arm_to_torso': 0.37,
            'golden_ratio': 1.618
        }
        
        if metric_type in ideal_values:
            deviation = abs(value - ideal_values[metric_type])
            return max(0, 1 - deviation)
        
        return 0

    def get_error_feedback(self, error_message):
        """Provide user-friendly feedback for common errors"""
        error_feedback = {
            'No pose landmarks detected': {
                'message': 'Could not detect body pose clearly',
                'suggestions': [
                    'Ensure full body is visible in the image',
                    'Wear form-fitting clothing',
                    'Improve lighting conditions',
                    'Stand against a contrasting background',
                    'Avoid loose or baggy clothing'
                ]
            },
            'No landmarks detected for measurements': {
                'message': 'Unable to take body measurements',
                'suggestions': [
                    'Stand straight with arms slightly away from body',
                    'Ensure good lighting',
                    'Wear clothing that shows body contours',
                    'Remove any objects blocking view of body'
                ]
            },
            'Image processing failed': {
                'message': 'Problem processing the image',
                'suggestions': [
                    'Check if image is clear and not blurry',
                    'Ensure image is not too dark or too bright',
                    'Try uploading in a different format (JPG/PNG)',
                    'Make sure image size is not too large or too small'
                ]
            }
        }
        
        for key in error_feedback:
            if key in error_message:
                return error_feedback[key]
                
        return {
            'message': 'An unexpected error occurred',
            'suggestions': [
                'Try uploading a different image',
                'Make sure the image shows your full body clearly',
                'Check image quality and lighting'
            ]
        } 