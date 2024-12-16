import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

class VideoThumbnailGenerator:
    def __init__(self):
        # Initialize models
        self.frame_classifier = pipeline("image-classification", model="microsoft/resnet-50")
        self.caption_generator = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        self.scene_detector = self._initialize_scene_detector()
    
    def _initialize_scene_detector(self):
        # Simple scene detection using pixel differences
        return lambda frames: np.mean(np.abs(np.diff(frames, axis=0)), axis=(1,2,3))
    
    def extract_candidate_frames(self, video_path, n_candidates=10):
        """Extract potential thumbnail frames based on scene changes and content analysis."""
        video = VideoFileClip(video_path)
        fps = video.fps
        duration = video.duration
        
        # Sample frames at regular intervals
        frame_indices = np.linspace(0, duration * fps, n_candidates * 2, dtype=int)
        frames = []
        
        for idx in frame_indices:
            frame = video.get_frame(idx / fps)
            frames.append(frame)
        
        # Convert to numpy array for processing
        frames = np.array(frames)
        
        # Detect scene changes
        scene_scores = self._initialize_scene_detector()(frames)
        
        # Get frames with significant changes
        threshold = np.percentile(scene_scores, 75)
        candidate_indices = np.where(scene_scores > threshold)[0]
        
        return frames[candidate_indices], candidate_indices / fps
    
    def analyze_frame_quality(self, frame):
        """Analyze frame for thumbnail suitability."""
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Get classification confidence
        classifications = self.frame_classifier(pil_image)
        
        # Basic image quality metrics
        brightness = np.mean(frame)
        contrast = np.std(frame)
        
        quality_score = (classifications[0]['score'] + 
                        normalize(brightness) + 
                        normalize(contrast)) / 3
        
        return quality_score
    
    def generate_caption(self, frame):
        """Generate a descriptive caption for the frame."""
        pil_image = Image.fromarray(frame)
        caption = self.caption_generator(pil_image)[0]['generated_text']
        return caption
    
    def process_video(self, video_path, num_thumbnails=4):
        """Main processing pipeline to generate thumbnails with captions."""
        # Extract candidate frames
        frames, timestamps = self.extract_candidate_frames(video_path)
        
        # Score each frame
        frame_scores = []
        for frame in frames:
            score = self.analyze_frame_quality(frame)
            frame_scores.append(score)
        
        # Select best frames
        best_indices = np.argsort(frame_scores)[-num_thumbnails:]
        best_frames = frames[best_indices]
        
        # Generate results
        results = []
        for frame in best_frames:
            caption = self.generate_caption(frame)
            results.append({
                'frame': frame,
                'caption': caption,
                'timestamp': timestamps[np.where(frames == frame)[0][0]]
            })
        
        return results
    
    def visualize_results(self, results):
        """Display thumbnails with captions in a grid."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, result in enumerate(results):
            axes[idx].imshow(result['frame'])
            axes[idx].set_title(result['caption'])
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig