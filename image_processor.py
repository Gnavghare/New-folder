import cv2
import numpy as np
import os
import mediapipe as mp
import math

class ImageProcessor:
    def __init__(self):
        self.target_size = (224, 224)  # Standard input size for many CNN models
        
        # Initialize MediaPipe Pose model for body analysis
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
    
    def process_image(self, image_path):
        """
        Process an image for body analysis:
        1. Read the image
        2. Convert to RGB
        3. Return the processed image
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert from BGR to RGB (OpenCV loads as BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to a standard size while maintaining aspect ratio
        h, w = image_rgb.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            if h > w:
                new_h, new_w = max_dim, int(w * max_dim / h)
            else:
                new_h, new_w = int(h * max_dim / w), max_dim
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        return image_rgb
    
    def enhance_image(self, image):
        """
        Apply image enhancement techniques:
        1. Contrast enhancement
        2. Noise reduction
        3. Sharpening
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel back with A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Apply slight Gaussian blur for noise reduction
        denoised = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def analyze_body(self, image_rgb):
        """
        Analyze the body in the image using MediaPipe Pose
        Returns body measurements and pose landmarks
        """
        # Get image dimensions
        height, width, _ = image_rgb.shape
        
        # Process the image with MediaPipe Pose
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {
                "success": False,
                "message": "No body detected in the image"
            }
        
        # Create a copy of the image for visualization
        annotated_image = image_rgb.copy()
        
        # Draw the pose landmarks on the image
        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Calculate body measurements
        body_measurements = self._calculate_body_measurements(landmarks, image_rgb.shape)
        
        # Calculate advanced body metrics
        advanced_metrics = self._calculate_advanced_metrics(landmarks, image_rgb.shape)
        
        # Merge measurements and advanced metrics
        body_measurements.update(advanced_metrics)
        
        return {
            "success": True,
            "landmarks": landmarks,
            "annotated_image": annotated_image,
            "body_measurements": body_measurements,
            "segmentation_mask": results.segmentation_mask if results.segmentation_mask is not None else None
        }
    
    def _calculate_body_measurements(self, landmarks, image_shape):
        """
        Calculate body measurements from pose landmarks
        """
        height, width, _ = image_shape
        
        # Helper function to calculate distance between landmarks
        def distance(lm1, lm2):
            return np.sqrt(
                (landmarks[lm1].x * width - landmarks[lm2].x * width) ** 2 +
                (landmarks[lm1].y * height - landmarks[lm2].y * height) ** 2
            )
        
        # Check if key landmarks are visible
        key_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        
        for lm in key_landmarks:
            if landmarks[lm].visibility < 0.5:
                # If key landmark is not visible, use default values
                return self._get_default_measurements()
        
        # Calculate shoulder width (distance between left and right shoulders)
        shoulder_width = distance(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                                 self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        
        # Calculate hip width (distance between left and right hips)
        hip_width = distance(self.mp_pose.PoseLandmark.LEFT_HIP.value,
                            self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        
        # Calculate waist approximation (midpoint between hips and shoulders)
        waist_width = (shoulder_width + hip_width) / 2
        
        # Calculate chest width (approximation based on shoulders)
        chest_width = shoulder_width * 1.1
        
        # Calculate body height (from head to ankle)
        body_height = distance(self.mp_pose.PoseLandmark.NOSE.value,
                              self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        
        # Calculate arm length (shoulder to wrist)
        left_arm_length = (
            distance(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value) +
            distance(self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        )
        
        right_arm_length = (
            distance(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value) +
            distance(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
        )
        
        # Calculate leg length (hip to ankle)
        left_leg_length = (
            distance(self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value) +
            distance(self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        )
        
        right_leg_length = (
            distance(self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value) +
            distance(self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        )
        
        # Calculate shoulder-to-hip ratio
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0
        
        # Determine body type based on shoulder-to-hip ratio
        body_type = self._determine_body_type(shoulder_hip_ratio)
        
        # Calculate BMI approximation (this is a rough estimate)
        # We use the ratio of width to height as a proxy
        bmi_proxy = (shoulder_width + hip_width) / (2 * body_height)
        bmi_category = self._determine_bmi_category(bmi_proxy)
        
        # Calculate body fat percentage approximation
        # This is a very rough estimate based on visual cues
        body_fat_percentage = self._estimate_body_fat(landmarks, shoulder_hip_ratio, bmi_proxy)
        
        return {
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "waist_width": waist_width,
            "chest_width": chest_width,
            "body_height": body_height,
            "left_arm_length": left_arm_length,
            "right_arm_length": right_arm_length,
            "left_leg_length": left_leg_length,
            "right_leg_length": right_leg_length,
            "shoulder_hip_ratio": shoulder_hip_ratio,
            "body_type": body_type,
            "bmi_proxy": bmi_proxy,
            "bmi_category": bmi_category,
            "body_fat_percentage": body_fat_percentage
        }
    
    def _get_default_measurements(self):
        """Return default measurements when landmarks are not visible"""
        return {
            "shoulder_width": 100,
            "hip_width": 90,
            "waist_width": 85,
            "chest_width": 110,
            "body_height": 170,
            "left_arm_length": 70,
            "right_arm_length": 70,
            "left_leg_length": 80,
            "right_leg_length": 80,
            "shoulder_hip_ratio": 1.1,
            "body_type": "Rectangle (Shoulders and hips similar width)",
            "bmi_proxy": 0.17,
            "bmi_category": "Normal weight",
            "body_fat_percentage": 20
        }
    
    def _calculate_advanced_metrics(self, landmarks, image_shape):
        """
        Calculate advanced body metrics like posture analysis, symmetry, and proportions
        """
        height, width, _ = image_shape
        
        # Helper function to calculate angle between three points
        def calculate_angle(a, b, c):
            a_x, a_y = landmarks[a].x * width, landmarks[a].y * height
            b_x, b_y = landmarks[b].x * width, landmarks[b].y * height
            c_x, c_y = landmarks[c].x * width, landmarks[c].y * height
            
            angle = math.degrees(math.atan2(c_y - b_y, c_x - b_x) - math.atan2(a_y - b_y, a_x - b_x))
            if angle < 0:
                angle += 360
            
            return angle
        
        # Check if key landmarks are visible
        key_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.NOSE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        
        for lm in key_landmarks:
            if landmarks[lm].visibility < 0.5:
                # If key landmark is not visible, use default values
                return self._get_default_advanced_metrics()
        
        # Posture analysis - check if shoulders are level
        left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height
        right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height
        shoulder_level_diff = abs(left_shoulder_y - right_shoulder_y)
        
        # Posture analysis - check if hips are level
        left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height
        right_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height
        hip_level_diff = abs(left_hip_y - right_hip_y)
        
        # Spine alignment - check if spine is straight
        nose_x = landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * width
        mid_shoulder_x = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width + 
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width) / 2
        mid_hip_x = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width + 
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width) / 2
        
        spine_alignment_diff = max(abs(nose_x - mid_shoulder_x), abs(mid_shoulder_x - mid_hip_x))
        
        # Body symmetry score (0-100, higher is better)
        symmetry_factors = [
            shoulder_level_diff / height,
            hip_level_diff / height,
            spine_alignment_diff / width
        ]
        symmetry_score = 100 - min(100, sum(symmetry_factors) * 1000)
        
        # Calculate body proportions
        # Golden ratio for ideal proportions is approximately 1.618
        # Calculate torso length (mid-shoulder to mid-hip)
        mid_shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + 
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
        mid_hip_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        torso_length = abs(mid_shoulder_y - mid_hip_y) * height
        
        # Calculate leg length (mid-hip to mid-ankle)
        mid_ankle_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y + 
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2
        leg_length = abs(mid_hip_y - mid_ankle_y) * height
        
        proportion_ratio = leg_length / torso_length if torso_length > 0 else 0
        
        # Determine posture quality
        posture_quality = self._determine_posture_quality(shoulder_level_diff, hip_level_diff, spine_alignment_diff, height, width)
        
        return {
            "posture_quality": posture_quality,
            "symmetry_score": symmetry_score,
            "proportion_ratio": proportion_ratio,
            "shoulder_level_diff": shoulder_level_diff,
            "hip_level_diff": hip_level_diff,
            "spine_alignment_diff": spine_alignment_diff
        }
    
    def _get_default_advanced_metrics(self):
        """Return default advanced metrics when landmarks are not visible"""
        return {
            "posture_quality": "Fair",
            "symmetry_score": 75.0,
            "proportion_ratio": 1.2,
            "shoulder_level_diff": 5,
            "hip_level_diff": 5,
            "spine_alignment_diff": 10
        }
    
    def _determine_posture_quality(self, shoulder_diff, hip_diff, spine_diff, height, width):
        """
        Determine posture quality based on alignment metrics
        """
        # Normalize differences relative to image dimensions
        norm_shoulder_diff = shoulder_diff / height
        norm_hip_diff = hip_diff / height
        norm_spine_diff = spine_diff / width
        
        # Calculate weighted score (lower is better)
        weighted_score = (norm_shoulder_diff * 0.4 + norm_hip_diff * 0.3 + norm_spine_diff * 0.3) * 100
        
        if weighted_score < 1:
            return "Excellent"
        elif weighted_score < 2:
            return "Good"
        elif weighted_score < 4:
            return "Fair"
        else:
            return "Poor"
    
    def _determine_body_type(self, shoulder_hip_ratio):
        """
        Determine body type based on shoulder-to-hip ratio
        """
        if shoulder_hip_ratio > 1.4:
            return "Inverted Triangle (Broad shoulders, narrow hips)"
        elif shoulder_hip_ratio < 0.9:
            return "Pear (Narrow shoulders, wide hips)"
        elif 1.2 <= shoulder_hip_ratio <= 1.4:
            return "Athletic (Broad shoulders, proportionate hips)"
        else:
            return "Rectangle (Shoulders and hips similar width)"
    
    def _determine_bmi_category(self, bmi_proxy):
        """
        Determine BMI category based on proxy value
        Note: This is a rough approximation and not medically accurate
        """
        # These thresholds are arbitrary and would need calibration
        if bmi_proxy < 0.15:
            return "Underweight"
        elif 0.15 <= bmi_proxy < 0.18:
            return "Normal weight"
        elif 0.18 <= bmi_proxy < 0.22:
            return "Overweight"
        else:
            return "Obese"
    
    def _estimate_body_fat(self, landmarks, shoulder_hip_ratio, bmi_proxy):
        """
        Estimate body fat percentage based on visual cues
        Note: This is a very rough estimate and not medically accurate
        """
        # Base estimate on BMI proxy
        base_estimate = bmi_proxy * 100
        
        # Adjust based on shoulder-hip ratio (higher ratio often indicates more muscle)
        if shoulder_hip_ratio > 1.2:
            base_estimate -= 5
        
        # Clamp to reasonable range
        return max(5, min(40, base_estimate))
    
    def get_body_analysis_image(self, analysis_result):
        """
        Create a visualization of the body analysis
        """
        if not analysis_result["success"]:
            return None
        
        annotated_image = analysis_result["annotated_image"]
        measurements = analysis_result["body_measurements"]
        
        # Add text with measurements
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        
        # Create a semi-transparent overlay for text background
        h, w = annotated_image.shape[:2]
        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (0, 0), (300, 250), (0, 0, 0), -1)
        alpha = 0.7
        annotated_image = cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0)
        
        # Add measurements text
        y_pos = 20
        line_height = 20
        
        cv2.putText(annotated_image, f"Body Type: {measurements['body_type']}", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"BMI Category: {measurements['bmi_category']}", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Est. Body Fat: {measurements['body_fat_percentage']:.1f}%", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Posture Quality: {measurements['posture_quality']}", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Symmetry Score: {measurements['symmetry_score']:.1f}/100", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Shoulder/Hip Ratio: {measurements['shoulder_hip_ratio']:.2f}", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Shoulder Width: {measurements['shoulder_width']:.1f} px", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Hip Width: {measurements['hip_width']:.1f} px", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Waist Width: {measurements['waist_width']:.1f} px", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        cv2.putText(annotated_image, f"Chest Width: {measurements['chest_width']:.1f} px", 
                   (10, y_pos), font, font_scale, color, thickness)
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR) 