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

    def generate_caption(self, frame):
        """
        Generate caption for a frame using the transformer model
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            # Generate caption
            captions = self.caption_generator(pil_image)
            return captions[0]["generated_text"]
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "No caption generated"

    def process_video(self, video_path, num_thumbnails=4):
        """
        Main processing pipeline
        """
        print("Starting video processing...")

        # Extract frames
        frames, timestamps = self.extract_frames(
            video_path, n_frames=num_thumbnails * 2
        )

        if not frames:
            raise ValueError("No frames were extracted from the video")

        # Analyze and score frames
        results = []
        for frame, timestamp in zip(frames, timestamps):
            quality_score = self.analyze_frame_quality(frame)
            caption = self.generate_caption(frame)

            results.append(
                {
                    "frame": frame,
                    "timestamp": timestamp,
                    "quality_score": quality_score,
                    "caption": caption,
                }
            )

        # Sort by quality score and take the best ones
        results.sort(key=lambda x: x["quality_score"], reverse=True)
        best_results = results[:num_thumbnails]

        print(f"Generated {len(best_results)} thumbnails successfully")
        return best_results

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