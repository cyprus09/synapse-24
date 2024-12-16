import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import torch
import os
from datetime import timedelta


class VideoThumbnailGenerator:
    def __init__(self):
        print("Initializing models...")
        # Initialize the caption generator
        self.caption_generator = pipeline(
            "image-to-text", model="nlpconnect/vit-gpt2-image-captioning"
        )
        print("Models initialized successfully")

    def extract_frames(self, video_path, n_frames=10):
        """
        Extract frames from video using OpenCV
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Processing video: {video_path}")
        frames = []
        timestamps = []

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        print(f"Video duration: {duration:.2f} seconds")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")

        # Calculate frame intervals
        interval = total_frames // n_frames

        for frame_idx in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(frame_idx / fps)
                print(f"Extracted frame at {frame_idx / fps:.2f} seconds")
            else:
                print(f"Failed to extract frame at index {frame_idx}")

        cap.release()
        return frames, timestamps

    def analyze_frame_quality(self, frame):
        """
        Analyze frame quality using basic metrics
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Calculate basic metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize metrics
        brightness_score = brightness / 255.0  # 0-1 scale
        contrast_score = min(contrast / 128.0, 1.0)  # 0-1 scale
        blur_score = min(blur / 1000.0, 1.0)  # 0-1 scale

        # Combine scores
        quality_score = (brightness_score + contrast_score + blur_score) / 3.0

        return quality_score
    
    def analyze_brand_positions(self, frame):
        """
        Analyze potential brand/logo positions in the frame
        """
        height, width = frame.shape[:2]
        
        # Define regions of interest (ROIs) for brand placement
        rois = {
            'top_right': frame[0:height//4, 3*width//4:width],
            'bottom_right': frame[3*height//4:height, 3*width//4:width],
            'bottom_left': frame[3*height//4:height, 0:width//4]
        }
        
        max_score = 0
        for roi_name, roi in rois.items():
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Calculate metrics that might indicate presence of a brand/logo
            variance = np.var(gray_roi)
            edges = cv2.Canny(gray_roi, 100, 200)
            edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
            
            # Combine metrics
            roi_score = (variance / 10000 + edge_density) / 2
            max_score = max(max_score, roi_score)
        
        return max_score

    def process_video(self, video_path, num_thumbnails=4):
        """
        Main processing pipeline with face detection and position scoring
        """
        print("Starting video processing...")
        
        # Initialize face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Extract frames
        frames, timestamps = self.extract_frames(video_path, n_frames=num_thumbnails * 2)
        
        if not frames:
            raise ValueError("No frames were extracted from the video")
        
        # Analyze and score frames
        results = []
        for frame, timestamp in zip(frames, timestamps):
            # Basic quality score
            quality_score = self.analyze_frame_quality(frame)
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Calculate face score
            face_score = 0
            if len(faces) > 0:
                # Bonus for having faces
                face_score = 0.3  # Base score for having faces
                
                # Additional scoring based on face position and size
                for (x, y, w, h) in faces:
                    face_center_x = x + w/2
                    face_center_y = y + h/2
                    
                    # Prefer faces in the center third of the image
                    frame_height, frame_width = frame.shape[:2]
                    center_score = 1.0 - (abs(face_center_x - frame_width/2) / frame_width + 
                                        abs(face_center_y - frame_height/2) / frame_height) / 2
                    
                    # Prefer larger faces (but not too large)
                    face_size_ratio = (w * h) / (frame_width * frame_height)
                    size_score = 1.0 if 0.1 <= face_size_ratio <= 0.4 else 0.5
                    
                    face_score += (center_score + size_score) * 0.15
            
            # Position scoring for brand/logo placement (assuming top-right or bottom-right corners)
            # This examines edge regions for potential brand elements
            position_score = self.analyze_brand_positions(frame)
            
            # Calculate final score
            final_score = (quality_score * 0.4 +  # Base quality
                        face_score * 0.4 +      # Face presence and position
                        position_score * 0.2)    # Brand position
            
            caption = self.generate_caption(frame)
            
            results.append({
                "frame": frame,
                "timestamp": timestamp,
                "quality_score": final_score,
                "caption": caption,
                "has_faces": len(faces) > 0,
                "num_faces": len(faces),
                "brand_score": position_score
            })
        
        # Sort by final score and take the best ones
        results.sort(key=lambda x: x["quality_score"], reverse=True)
        best_results = results[:num_thumbnails]
        
        print(f"Generated {len(best_results)} thumbnails successfully")
        for idx, result in enumerate(best_results):
            print(f"Thumbnail {idx + 1}:")
            print(f"  - Quality Score: {result['quality_score']:.2f}")
            print(f"  - Number of Faces: {result['num_faces']}")
            print(f"  - Brand Score: {result['brand_score']:.2f}")
            print(f"  - Caption: {result['caption']}")
        
        return best_results

    def enhance_caption(self, base_caption):
        """
        Transform basic image descriptions into engaging video-style captions
        """
        # Keywords that suggest action
        action_words = {
            'standing': 'Discover',
            'sitting': 'Experience',
            'looking': 'Explore',
            'walking': 'Journey through',
            'talking': 'Connect with',
            'eating': 'Savor',
            'working': 'Master',
            'playing': 'Enjoy'
        }
        
        # Keywords for locations/settings
        setting_emphasis = {
            'restaurant': 'finest dining spots',
            'office': 'professional workspace',
            'street': 'urban adventure',
            'park': 'natural getaway',
            'beach': 'coastal paradise',
            'gym': 'fitness journey',
            'house': 'perfect space',
            'room': 'intimate setting'
        }
        
        # Convert basic description to engaging caption
        caption = base_caption.lower()
        
        # Replace basic action words with engaging ones
        for action, replacement in action_words.items():
            if action in caption:
                caption = caption.replace(f"is {action}", replacement)
                break
        
        # Enhance location descriptions
        for setting, enhanced in setting_emphasis.items():
            if setting in caption:
                caption = caption.replace(setting, enhanced)
                break
        
        # Remove common basic phrases
        phrases_to_remove = [
            "there is ", "there are ",
            "this is ", "these are ",
            "a person", "some people",
            "a woman", "a man"
        ]
        for phrase in phrases_to_remove:
            caption = caption.replace(phrase, "")
        
        # Add engaging prefixes if caption seems too plain
        engaging_prefixes = [
            "Discover the Magic:",
            "Experience the Moment:",
            "Journey Inside:",
            "Exclusive Look:",
            "Behind the Scenes:",
            "Captivating Views:",
            "Unforgettable Moments:",
            "Live the Experience:"
        ]
        
        if len(caption) < 30 or caption.startswith(('a ', 'the ', 'an ')):
            import random
            caption = f"{random.choice(engaging_prefixes)} {caption}"
        
        # Capitalize first letter of each word for title-style caption
        caption = ' '.join(word.capitalize() for word in caption.split())
        
        # Add engaging suffixes for certain types of content
        if "people" in caption.lower() or "together" in caption.lower():
            caption += " | Creating Memories That Last"
        elif any(word in caption.lower() for word in ["food", "drink", "dining"]):
            caption += " | A Taste of Excellence"
        elif any(word in caption.lower() for word in ["view", "scene", "landscape"]):
            caption += " | Breathtaking Moments"
            
        return caption

    def get_caption_style(self, frame_analysis):
        """
        Determine appropriate caption style based on frame content
        """
        styles = {
            'action': {
                'prefix': ['Watch', 'Experience', 'Discover'],
                'suffix': ['in Action', 'Like Never Before', 'at Its Best']
            },
            'scenic': {
                'prefix': ['Explore', 'Journey Through', 'Discover'],
                'suffix': ['in All Its Glory', 'at Its Finest', 'Like Never Before']
            },
            'emotional': {
                'prefix': ['Feel', 'Share', 'Connect with'],
                'suffix': ['from the Heart', 'that Touches the Soul', 'that Inspires']
            }
        }
        
        # Analyze frame content to determine style
        # This could be enhanced with more sophisticated image analysis
        return styles['action']  # Default style for now

    def generate_caption(self, frame):
        """
        Generate engaging video-style captions by combining image analysis with caption enhancement
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Get base caption from the model
            base_caption = self.caption_generator(pil_image)[0]['generated_text']
            
            # Transform the caption into a more engaging format
            enhanced_caption = self.enhance_caption(base_caption)
            
            return enhanced_caption
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "Exciting moment from the video"

    def save_thumbnails(self, results, output_dir="thumbnails"):
        """
        Save thumbnails and their captions to disk
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, result in enumerate(results):
            # Save image
            img_path = os.path.join(output_dir, f"thumbnail_{i}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(result["frame"], cv2.COLOR_RGB2BGR))

            # Save caption
            caption_path = os.path.join(output_dir, f"thumbnail_{i}_caption.txt")
            with open(caption_path, "w") as f:
                f.write(f"Caption: {result['caption']}\n")
                f.write(f"Timestamp: {timedelta(seconds=int(result['timestamp']))}\n")
                f.write(f"Quality Score: {result['quality_score']:.2f}\n")

            print(f"Saved thumbnail {i+1} to {img_path}")