import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import mediapipe as mp
import os

class ImageUtils:
    @staticmethod
    def enhance_image_for_recognition(image):
        """
        Apply multiple enhancement techniques to improve image quality for recognition
        """
        try:
            # Convert to PIL Image if necessary
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhance brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Convert back to numpy array
            image = np.array(image)
            
            return image
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return image

    @staticmethod
    def remove_background(image):
        """
        Remove background from image using MediaPipe Selfie Segmentation
        """
        try:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
            
            # Convert to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            results = selfie_segmentation.process(image)
            
            # Create white background
            bg_image = np.ones(image.shape, dtype=np.uint8) * 255
            
            # Generate segmentation mask
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            
            # Merge foreground and background
            output_image = np.where(condition, image, bg_image)
            
            return output_image
        except Exception as e:
            print(f"Error removing background: {str(e)}")
            return image

    @staticmethod
    def normalize_image_size(image, target_size=(512, 512)):
        """
        Normalize image size while preserving aspect ratio
        """
        try:
            h, w = image.shape[:2]
            aspect = w / h
            
            if aspect > 1:
                # Width is greater than height
                new_w = target_size[0]
                new_h = int(new_w / aspect)
            else:
                # Height is greater than width
                new_h = target_size[1]
                new_w = int(new_h * aspect)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create white canvas of target size
            canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
            
            # Calculate position to paste resized image
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            
            # Paste resized image
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        except Exception as e:
            print(f"Error normalizing image size: {str(e)}")
            return image

    @staticmethod
    def apply_lighting_correction(image):
        """
        Apply lighting correction to improve image quality
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl,a,b))
            
            # Convert back to RGB
            corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            return corrected
        except Exception as e:
            print(f"Error correcting lighting: {str(e)}")
            return image

    @staticmethod
    def detect_blur(image):
        """
        Detect if image is blurry using Laplacian variance
        Returns: (is_blurry, blur_score)
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = np.var(laplacian)
            
            # Threshold for blur detection
            threshold = 100
            is_blurry = score < threshold
            
            return is_blurry, score
        except Exception as e:
            print(f"Error detecting blur: {str(e)}")
            return True, 0

    @staticmethod
    def check_image_quality(image):
        """
        Check various image quality metrics
        Returns: dict with quality metrics
        """
        try:
            quality_metrics = {
                'resolution': image.shape[:2],
                'aspect_ratio': image.shape[1] / image.shape[0],
                'is_blurry': ImageUtils.detect_blur(image)[0],
                'blur_score': ImageUtils.detect_blur(image)[1],
                'brightness': np.mean(image),
                'contrast': np.std(image)
            }
            
            # Add quality assessment
            quality_metrics['is_good_quality'] = (
                not quality_metrics['is_blurry'] and
                quality_metrics['brightness'] > 50 and
                quality_metrics['brightness'] < 200 and
                quality_metrics['contrast'] > 30
            )
            
            return quality_metrics
        except Exception as e:
            print(f"Error checking image quality: {str(e)}")
            return None

    @staticmethod
    def prepare_image_for_analysis(image_path):
        """
        Prepare image for analysis by applying all necessary preprocessing steps
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check quality
            quality_metrics = ImageUtils.check_image_quality(image)
            if quality_metrics and not quality_metrics['is_good_quality']:
                # Apply enhancements if quality is not good
                image = ImageUtils.enhance_image_for_recognition(image)
                image = ImageUtils.apply_lighting_correction(image)
            
            # Remove background
            image = ImageUtils.remove_background(image)
            
            # Normalize size
            image = ImageUtils.normalize_image_size(image)
            
            return image, quality_metrics
        except Exception as e:
            print(f"Error preparing image: {str(e)}")
            return None, None

    @staticmethod
    def validate_image(image_path):
        """
        Validate if image is suitable for body analysis
        Returns: (is_valid, message)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, "Could not read image file"
            
            # Check file size
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
            if file_size > 10:
                return False, "Image file size too large (max 10MB)"
            
            # Check dimensions
            height, width = image.shape[:2]
            if height < 300 or width < 300:
                return False, "Image resolution too low (min 300x300)"
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                return False, "Invalid aspect ratio (should be between 0.5 and 2.0)"
            
            # Check quality metrics
            quality_metrics = ImageUtils.check_image_quality(image)
            if quality_metrics:
                if quality_metrics['is_blurry']:
                    return False, "Image is too blurry"
                if quality_metrics['brightness'] < 50:
                    return False, "Image is too dark"
                if quality_metrics['brightness'] > 200:
                    return False, "Image is too bright"
                if quality_metrics['contrast'] < 30:
                    return False, "Image has insufficient contrast"
            
            return True, "Image is valid for analysis"
        except Exception as e:
            return False, f"Error validating image: {str(e)}" 