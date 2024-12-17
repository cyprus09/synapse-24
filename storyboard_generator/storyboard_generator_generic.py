from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple
from googleapiclient.discovery import build
import librosa
import soundfile as sf
import tempfile
import subprocess
import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import logging
import requests
from io import BytesIO


@dataclass
class ContentMetrics:
    views: int
    likes: int
    comments: int
    engagement_rate: float
    duration: float  # in seconds
    tags: List[str]
    category_id: str
    title: str
    description: str
    publish_date: datetime
    thumbnail_url: str


@dataclass
class AudioAnalysis:
    average_volume: float
    peak_volume: float
    speech_segments: List[Tuple[float, float]]
    audio_quality: Dict[str, float]
    tempo: float
    background_noise: float


@dataclass
class ContentAnalysis:
    duration: float
    resolution: Tuple[int, int]
    fps: float
    brightness_analysis: Dict[str, float]
    scene_changes: List[float]
    thumbnail_data: Dict[str, float]
    audio_analysis: Optional[AudioAnalysis] = None


@dataclass
class ContentRecommendations:
    key_improvements: List[Dict[str, str]]
    title_suggestions: List[str]
    thumbnail_improvements: List[str]
    technical_optimizations: List[Dict[str, str]]
    engagement_strategies: List[Dict[str, str]]
    benchmark_learnings: List[Dict[str, str]]


@dataclass
class PredictiveMetrics:
    viral_probability: float
    estimated_views_30d: int
    estimated_engagement_rate: float
    best_posting_times: List[time]
    target_demographics: List[Dict[str, str]]
    content_virality_factors: Dict[str, float]
    viewer_retention_estimate: float
    recommended_hashtags: List[str]
    competition_level: str
    growth_potential: str
    benchmark_comparison: List[Dict[str, str]]


class StoryBoardGeneratorGeneric:
    def __init__(self, youtube_api_key: str):
        self.youtube = build("youtube", "v3", developerKey=youtube_api_key)
        self.logger = logging.getLogger(__name__)

    def analyze_youtube_video(self, video_id: str) -> ContentMetrics:
        """Analyze a YouTube video using only publicly available data via Youtube Data API"""
        try:
            # Get video details
            video_response = (
                self.youtube.videos()
                .list(part="snippet,statistics,contentDetails", id=video_id)
                .execute()
            )

            if not video_response["items"]:
                raise ValueError(f"Video {video_id} not found")

            video_data = video_response["items"][0]
            snippet = video_data["snippet"]
            stats = video_data["statistics"]

            # Parse duration from contentDetails
            duration_str = video_data["contentDetails"]["duration"]
            duration = self._parse_duration(duration_str)

            return ContentMetrics(
                views=int(stats.get("viewCount", 0)),
                likes=int(stats.get("likeCount", 0)),
                comments=int(stats.get("commentCount", 0)),
                engagement_rate=self._calculate_engagement_rate(stats),
                duration=duration,
                tags=snippet.get("tags", []),
                category_id=snippet.get("categoryId", ""),
                title=snippet.get("title", ""),
                description=snippet.get("description", ""),
                publish_date=datetime.strptime(
                    snippet.get("publishedAt"), "%Y-%m-%dT%H:%M:%SZ"
                ),
                thumbnail_url=snippet.get("thumbnails", {})
                .get("high", {})
                .get("url", ""),
            )

        except Exception as e:
            self.logger.error(f"Error analyzing YouTube video: {str(e)}")
            raise

    def analyze_local_video(self, video_path: str) -> ContentAnalysis:
        """Analyze a local video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get basic video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize analysis containers
            brightness_values = []
            scene_changes = []
            prev_frame = None

            # Sample frames at regular intervals
            sample_rate = max(1, total_frames // 100)  # Analyze up to 100 frames

            for frame_no in range(0, total_frames, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()

                if ret:
                    # Brightness analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_values.append(np.mean(gray))

                    # Scene change detection
                    if prev_frame is not None:
                        diff = cv2.absdiff(prev_frame, gray)
                        scene_changes.append(np.mean(diff))

                    prev_frame = gray.copy()

            cap.release()

            # Extract and analyze audio
            audio_path = self._extract_audio(video_path)
            audio_analysis = self._analyze_audio(audio_path)
            os.unlink(audio_path)  # Clean up temporary audio file

            return ContentAnalysis(
                duration=total_frames / fps,
                resolution=(width, height),
                fps=fps,
                brightness_analysis={
                    "mean": float(np.mean(brightness_values)),
                    "std": float(np.std(brightness_values)),
                    "min": float(np.min(brightness_values)),
                    "max": float(np.max(brightness_values)),
                },
                scene_changes=[float(x) for x in scene_changes],
                thumbnail_data=self._analyze_first_frame(video_path),
                audio_analysis=audio_analysis,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing local video: {str(e)}")
            raise

    def compare_videos(self, youtube_metrics, local_analysis) -> ContentRecommendations:
        """Generate recommendations for local video based on benchmark comparison"""
        try:
            # Analyze title effectiveness
            title_suggestions = self._analyze_title(youtube_metrics.title)

            # Generate key improvements based on benchmark comparison
            key_improvements = self._identify_key_improvements(
                local_analysis, youtube_metrics
            )

            # Technical optimizations based on quality comparison
            technical_optimizations = self._generate_technical_optimizations(
                local_analysis, youtube_metrics
            )

            # Analyze successful elements from benchmark
            benchmark_learnings = self._extract_benchmark_learnings(
                local_analysis, youtube_metrics
            )

            # Generate engagement strategies
            engagement_strategies = self._generate_engagement_strategies(
                local_analysis, youtube_metrics
            )

            # Thumbnail recommendations
            thumbnail_improvements = self._generate_thumbnail_recommendations(
                local_analysis.thumbnail_data
            )

            return ContentRecommendations(
                key_improvements=key_improvements,
                title_suggestions=title_suggestions,
                thumbnail_improvements=thumbnail_improvements,
                technical_optimizations=technical_optimizations,
                engagement_strategies=engagement_strategies,
                benchmark_learnings=benchmark_learnings,
            )

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def _identify_key_improvements(
        self, local_analysis, youtube_metrics
    ) -> List[Dict[str, str]]:
        """Identify key areas for improvement based on benchmark comparison"""
        improvements = []

        # Duration comparison
        duration_ratio = local_analysis.duration / youtube_metrics.duration
        if duration_ratio > 1.3:
            improvements.append(
                {
                    "area": "Video Length",
                    "issue": "Video is significantly longer than benchmark",
                    "recommendation": f"Consider trimming content to around {int(youtube_metrics.duration/60)} minutes",
                    "impact": "HIGH",
                }
            )
        elif duration_ratio < 0.7:
            improvements.append(
                {
                    "area": "Video Length",
                    "issue": "Video might be too short compared to benchmark",
                    "recommendation": "Add more valuable content or detailed explanations",
                    "impact": "MEDIUM",
                }
            )

        # Scene dynamics
        benchmark_scenes = max(
            youtube_metrics.duration / 10, 1
        )  # Assume 1 scene change every 10s as minimum
        local_scenes = len(local_analysis.scene_changes)
        if local_scenes < benchmark_scenes * 0.7:
            improvements.append(
                {
                    "area": "Visual Engagement",
                    "issue": "Lower scene variety compared to benchmark",
                    "recommendation": "Add more visual variety, transitions, or camera angles",
                    "impact": "HIGH",
                }
            )

        # Audio quality comparison
        if local_analysis.audio_analysis:
            if local_analysis.audio_analysis.background_noise > 0.2:
                improvements.append(
                    {
                        "area": "Audio Quality",
                        "issue": "Higher background noise than ideal",
                        "recommendation": "Use noise reduction, record in quieter environment, or use better microphone",
                        "impact": "HIGH",
                    }
                )

        return improvements

    def _generate_technical_optimizations(
        self, local_analysis, youtube_metrics
    ) -> List[Dict[str, str]]:
        """Generate technical optimization recommendations"""
        optimizations = []

        # Resolution check
        if local_analysis.resolution[0] < 1920:
            optimizations.append(
                {
                    "aspect": "Video Resolution",
                    "current": f"{local_analysis.resolution[0]}x{local_analysis.resolution[1]}",
                    "recommended": "1920x1080 or higher",
                    "reason": "Higher resolution improves viewer experience and perceived quality",
                }
            )

        # Frame rate analysis
        if local_analysis.fps < 30:
            optimizations.append(
                {
                    "aspect": "Frame Rate",
                    "current": f"{local_analysis.fps:.1f} FPS",
                    "recommended": "30 FPS or higher",
                    "reason": "Smoother playback enhances viewing experience",
                }
            )

        # Brightness and contrast
        if local_analysis.brightness_analysis["std"] < 40:
            optimizations.append(
                {
                    "aspect": "Visual Contrast",
                    "current": "Low contrast",
                    "recommended": "Increase lighting dynamics",
                    "reason": "Better contrast improves visual engagement",
                }
            )

        return optimizations

    def _extract_benchmark_learnings(
        self, local_analysis, youtube_metrics
    ) -> List[Dict[str, str]]:
        """Extract successful elements from benchmark video"""
        learnings = []

        # Title structure
        if youtube_metrics.title:
            learnings.append(
                {
                    "element": "Title Structure",
                    "observation": self._analyze_title_structure(youtube_metrics.title),
                    "application": "Apply similar pattern to your titles",
                }
            )

        # Tag usage
        if youtube_metrics.tags:
            learnings.append(
                {
                    "element": "Tag Strategy",
                    "observation": f"Uses {len(youtube_metrics.tags)} targeted tags",
                    "application": "Implement similar tag density and relevance",
                }
            )

        # Content pacing
        avg_scene_duration = youtube_metrics.duration / (
            len(local_analysis.scene_changes) + 1
        )
        learnings.append(
            {
                "element": "Content Pacing",
                "observation": f"Average scene duration: {avg_scene_duration:.1f} seconds",
                "application": "Match scene change frequency for optimal engagement",
            }
        )

        return learnings

    def _generate_engagement_strategies(
        self, local_analysis, youtube_metrics
    ) -> List[Dict[str, str]]:
        """Generate engagement strategies based on benchmark success"""
        strategies = []

        # Opening hook
        if local_analysis.duration > 0:
            strategies.append(
                {
                    "aspect": "Opening Hook",
                    "strategy": "Create compelling first 15 seconds",
                    "implementation": "Include key value proposition and visual interest early",
                    "priority": "HIGH",
                }
            )

        # Calls to action
        strategies.append(
            {
                "aspect": "Viewer Interaction",
                "strategy": "Strategic call-to-actions",
                "implementation": "Add subscription/like reminders at key moments",
                "priority": "MEDIUM",
            }
        )

        # Content structure
        strategies.append(
            {
                "aspect": "Content Structure",
                "strategy": "Clear content progression",
                "implementation": "Use chapters/segments like benchmark video",
                "priority": "HIGH",
            }
        )

        return strategies

    def _analyze_title_structure(self, title: str) -> str:
        """Analyze successful title structure"""
        patterns = []
        if ":" in title:
            patterns.append("Uses colon for subtitle")
        if "?" in title:
            patterns.append("Uses question format")
        if any(char.isupper() for char in title):
            patterns.append("Uses strategic capitalization")
        if any(str(num) for num in range(10) if str(num) in title):
            patterns.append("Includes numbers")

        return " | ".join(patterns) if patterns else "Simple, direct title"

    def _generate_thumbnail_recommendations(self, thumbnail_data: Dict) -> List[str]:
        """Generate thumbnail improvement recommendations"""
        recommendations = []

        if thumbnail_data:
            if thumbnail_data.get("brightness", 0) < 100:
                recommendations.append(
                    "Increase thumbnail brightness for better visibility"
                )
            if thumbnail_data.get("contrast", 0) < 50:
                recommendations.append("Improve thumbnail contrast for visual impact")
            if thumbnail_data.get("saturation", 0) < 100:
                recommendations.append("Consider increasing color saturation")

        recommendations.extend(
            [
                "Include clear, readable text overlay",
                "Use close-up shots of key subjects",
                "Implement color psychology for emotional impact",
            ]
        )

        return recommendations

    def _parse_duration(self, duration_str: str) -> float:
        hours = minutes = seconds = 0
        duration = duration_str[2:]

        if "H" in duration:
            hours, duration = duration.split("H")
            hours = int(hours)
        if "M" in duration:
            minutes, duration = duration.split("M")
            minutes = int(minutes)
        if "S" in duration:
            seconds = int(duration.replace("S", ""))

        return float(hours * 3600 + minutes * 60 + seconds)

    def _calculate_engagement_rate(self, stats: Dict) -> float:
        """Calculate engagement rate using available metrics"""
        try:
            views = int(stats.get("viewCount", 0))
            if views == 0:
                return 0.0

            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            # Basic engagement calculation
            engagement = (likes + comments) / views
            return round(engagement * 100, 2)
        except Exception as e:
            self.logger.warning(f"Error calculating engagement rate: {str(e)}")
            return 0.0

    def _analyze_title(self, title: str) -> List[str]:
        """Analyze title and suggest improvements"""
        suggestions = []

        if len(title) < 30:
            suggestions.append(
                "Consider a longer title (30-60 characters) for better SEO"
            )
        if len(title) > 60:
            suggestions.append("Title might be too long - consider shortening it")
        if not any(char in title for char in "?!💡✨🔥"):
            suggestions.append(
                "Consider adding emojis or special characters for better CTR"
            )

        return suggestions

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            command = [
                "ffmpeg",
                "-i",
                video_path,
                "-ab",
                "160k",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-vn",
                temp_audio.name,
                "-y",
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise Exception(f"Failed to extract audio: {stderr.decode()}")

            return temp_audio.name

        except Exception as e:
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            raise Exception(f"Error extracting audio: {str(e)}")

    def _analyze_audio(self, audio_path: str) -> AudioAnalysis:
        # Perform detailed audio analysis
        try:
            y, sr = librosa.load(audio_path)

            # Calculate volume metrics
            rms = librosa.feature.rms(y=y)[0]
            average_volume = float(np.mean(rms))
            peak_volume = float(np.max(rms))

            # Detect speech segments using energy threshold
            speech_threshold = np.mean(rms) * 1.2
            speech_frames = rms > speech_threshold
            speech_segments = []
            current_segment = None

            for i, is_speech in enumerate(speech_frames):
                time = librosa.frames_to_time(i, sr=sr)

                if is_speech and current_segment is None:
                    current_segment = time
                elif not is_speech and current_segment is not None:
                    speech_segments.append((float(current_segment), float(time)))
                    current_segment = None

            # Get tempo
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])

            # Calculate spectral features for quality assessment
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

            # Estimate background noise using percentile
            background_noise = float(np.percentile(rms, 10))

            # Calculate audio quality metrics
            audio_quality = {
                "clarity": float(np.mean(spectral_centroid) / sr),
                "fullness": float(np.mean(spectral_rolloff) / sr),
                "variation": float(
                    np.std(spectral_bandwidth) / np.mean(spectral_bandwidth)
                ),
                "signal_to_noise": float(
                    average_volume / background_noise if background_noise > 0 else 0
                ),
            }

            return AudioAnalysis(
                average_volume=average_volume,
                peak_volume=peak_volume,
                speech_segments=speech_segments,
                audio_quality=audio_quality,
                tempo=tempo,
                background_noise=background_noise,
            )

        except Exception as e:
            self.logger.error(f"Error in audio analysis: {str(e)}")
            return None

    def _analyze_first_frame(self, video_path: str) -> Dict[str, float]:
        """Analyze the first frame of the video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return {}

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            return {
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray)),
                "saturation": float(np.mean(hsv[:, :, 1])),
                "colorfulness": float(np.std(hsv[:, :, 0])),
            }
        except Exception as e:
            self.logger.warning(f"Could not analyze first frame: {str(e)}")
            return {}

    def _generate_content_improvements(
        self, local_analysis: ContentAnalysis, youtube_metrics: ContentMetrics
    ) -> List[Dict[str, str]]:
        """Generate content improvement suggestions based on analysis"""
        improvements = []

        # Compare durations
        if local_analysis.duration > youtube_metrics.duration * 1.5:
            improvements.append(
                {
                    "area": "Duration",
                    "suggestion": "Video might be too long compared to reference",
                    "action": "Consider trimming content to maintain viewer engagement",
                }
            )

        # Analyze scene changes
        avg_scene_duration = local_analysis.duration / (
            len(local_analysis.scene_changes) + 1
        )
        if avg_scene_duration > 10:
            improvements.append(
                {
                    "area": "Pacing",
                    "suggestion": "Scenes might be too long",
                    "action": "Consider adding more scene transitions to maintain viewer interest",
                }
            )

        # Analyze audio if available
        if local_analysis.audio_analysis:
            if local_analysis.audio_analysis.background_noise > 0.1:
                improvements.append(
                    {
                        "area": "Audio",
                        "suggestion": "High background noise detected",
                        "action": "Consider using noise reduction or recording in a quieter environment",
                    }
                )

        return improvements

    def _generate_optimizations(
        self, local_analysis: ContentAnalysis, youtube_metrics: ContentMetrics
    ) -> List[str]:
        """Generate optimization suggestions"""
        optimizations = []

        # Resolution optimizations
        width, height = local_analysis.resolution
        if width < 1920 or height < 1080:
            optimizations.append(
                "Consider recording in at least 1080p for better quality"
            )

        # FPS optimizations
        if local_analysis.fps < 30:
            optimizations.append(
                "Consider recording at 30fps or higher for smoother playback"
            )

        # Brightness optimizations
        if local_analysis.brightness_analysis["std"] < 30:
            optimizations.append(
                "Low contrast detected - consider adding more dynamic lighting"
            )

        return optimizations

    def predict_performance(self, youtube_metrics, local_analysis) -> PredictiveMetrics:
        """
        Predict local video performance using YouTube video as benchmark
        """
        # Compare local video quality against benchmark
        benchmark_comparison = {
            "resolution": self._compare_resolution(local_analysis, youtube_metrics),
            "duration": self._compare_duration(local_analysis, youtube_metrics),
            "pacing": self._analyze_pacing(local_analysis.scene_changes),
            "audio_quality": self._analyze_audio_engagement(
                local_analysis.audio_analysis
            ),
            "production_value": self._calculate_quality_score(local_analysis),
        }

        # Calculate overall quality ratio compared to benchmark
        quality_ratio = sum(benchmark_comparison.values()) / len(benchmark_comparison)

        # Analyze content virality factors for local video
        viral_factors = {
            "production_quality": benchmark_comparison["production_value"],
            "content_pacing": benchmark_comparison["pacing"],
            "audio_quality": benchmark_comparison["audio_quality"],
            "engagement_potential": self._estimate_engagement_potential(
                local_analysis, youtube_metrics
            ),
        }

        # Calculate viral probability based on benchmark comparison
        viral_probability = self._calculate_viral_probability(
            viral_factors, youtube_metrics
        )

        # Estimate potential views based on benchmark performance
        potential_views = self._estimate_potential_views(youtube_metrics, quality_ratio)

        # Analyze competitive advantages/disadvantages
        competitive_advantage = self._analyze_competitive_advantage(
            local_analysis, youtube_metrics
        )

        # Calculate potential engagement rate
        potential_engagement = self._estimate_potential_engagement(
            youtube_metrics, viral_factors
        )

        # Estimate retention based on local video characteristics
        retention_estimate = self._estimate_retention(local_analysis, viral_factors)

        return PredictiveMetrics(
            viral_probability=viral_probability,
            estimated_views_30d=potential_views,
            estimated_engagement_rate=potential_engagement,
            best_posting_times=self._calculate_optimal_posting_times(
                youtube_metrics.category_id
            ),
            target_demographics=self._analyze_target_demographics(
                youtube_metrics, local_analysis
            ),
            content_virality_factors=viral_factors,
            viewer_retention_estimate=retention_estimate,
            recommended_hashtags=self._generate_hashtags(
                youtube_metrics, local_analysis
            ),
            competition_level=competitive_advantage,
            growth_potential=self._calculate_growth_potential(
                viral_factors, competitive_advantage
            ),
            benchmark_comparison=benchmark_comparison,
        )

    def _compare_resolution(self, local_analysis, youtube_metrics) -> float:
        """Compare local video resolution to typical successful videos"""
        local_res = local_analysis.resolution[0] * local_analysis.resolution[1]
        benchmark_res = 1920 * 1080  # HD benchmark
        return min(local_res / benchmark_res, 1.0)

    def _compare_duration(self, local_analysis, youtube_metrics) -> float:
        """Compare video duration effectiveness"""
        # Calculate optimal duration ratio (not too short, not too long compared to benchmark)
        duration_ratio = local_analysis.duration / youtube_metrics.duration
        if duration_ratio > 1.5 or duration_ratio < 0.5:
            return 0.5
        return 1.0 - abs(1 - duration_ratio)

    def _calculate_optimal_posting_times(self, category_id: str) -> List[time]:
        """Determine best posting times based on category"""
        # Mapping of category IDs to optimal posting times
        category_times = {
            # Gaming
            "20": [time(15, 0), time(18, 0), time(21, 0)],
            # Entertainment
            "24": [time(12, 0), time(15, 0), time(19, 0)],
            # How-to & Style
            "26": [time(10, 0), time(14, 0), time(17, 0)],
            # Default times
            "default": [time(11, 0), time(15, 0), time(19, 0)],
        }
        return category_times.get(category_id, category_times["default"])

    def _analyze_pacing(self, scene_changes: List[float]) -> float:
        """Analyze content pacing based on scene changes"""
        if not scene_changes:
            return 0.5

        avg_scene_duration = len(scene_changes) / len(scene_changes)
        return min(1.0, 1.5 / avg_scene_duration) if avg_scene_duration > 0 else 0.5

    def _calculate_quality_score(self, analysis) -> float:
        """Calculate overall production quality score"""
        scores = {
            "resolution": min(analysis.resolution[0] / 1920, 1),  # normalize to 1080p
            "framerate": min(analysis.fps / 30, 1),  # normalize to 30fps
            "brightness": min(
                analysis.brightness_analysis["std"] / 50, 1
            ),  # dynamic range
            "scene_variety": min(
                len(analysis.scene_changes) / (analysis.duration / 30), 1
            ),
        }
        return sum(scores.values()) / len(scores)

    def _estimate_engagement_potential(self, local_analysis, youtube_metrics) -> float:
        """Estimate potential engagement based on content characteristics"""
        factors = {
            "pacing_score": self._analyze_pacing(local_analysis.scene_changes),
            "audio_engagement": self._analyze_audio_engagement(
                local_analysis.audio_analysis
            ),
            "visual_interest": len(local_analysis.scene_changes)
            / local_analysis.duration,
        }
        return sum(factors.values()) / len(factors)

    def _analyze_audio_engagement(self, audio_analysis) -> float:
        """Analyze audio characteristics for engagement potential"""
        if not audio_analysis:
            return 0.5

        engagement_factors = {
            "clarity": audio_analysis.audio_quality.get("clarity", 0.5),
            "dynamics": 1
            - abs(audio_analysis.peak_volume - audio_analysis.average_volume),
            "speech_presence": len(audio_analysis.speech_segments)
            / 60,  # segments per minute
        }
        return sum(engagement_factors.values()) / len(engagement_factors)

    def _analyze_competitive_advantage(
        self, local_analysis, youtube_metrics
    ) -> Dict[str, str]:
        """Analyze competitive advantages and disadvantages"""
        advantages = {}

        # Resolution comparison
        if local_analysis.resolution[0] >= 1920:
            advantages["resolution"] = "Higher quality video resolution"

        # Audio quality comparison
        if (
            local_analysis.audio_analysis
            and local_analysis.audio_analysis.background_noise < 0.1
        ):
            advantages["audio"] = "Superior audio quality"

        # Content pacing
        scene_frequency = len(local_analysis.scene_changes) / local_analysis.duration
        if scene_frequency > 0.2:  # More than 1 scene change every 5 seconds
            advantages["pacing"] = "Dynamic content pacing"

        # Production value
        if self._calculate_quality_score(local_analysis) > 0.8:
            advantages["production"] = "High production value"

        if not advantages:
            advantages["note"] = (
                "Consider improving production quality to match benchmark"
            )

        return advantages

    def _calculate_viral_probability(
        self, viral_factors: Dict[str, float], youtube_metrics
    ) -> float:
        """Calculate probability of video going viral based on benchmark comparison"""
        base_probability = sum(viral_factors.values()) / len(viral_factors)

        # Adjust based on benchmark success
        benchmark_multiplier = min(youtube_metrics.engagement_rate / 5, 2)

        return min(base_probability * benchmark_multiplier * 100, 100)

    def _estimate_potential_views(self, youtube_metrics, quality_ratio: float) -> int:
        """Estimate potential views based on benchmark performance and quality"""
        base_views = youtube_metrics.views
        conservative_estimate = int(
            base_views * quality_ratio * 0.7
        )  # 70% of benchmark as base
        return max(conservative_estimate, 1000)  # Minimum threshold

    def _estimate_retention(
        self, local_analysis, viral_factors: Dict[str, float]
    ) -> float:
        """Estimate viewer retention rate"""
        base_retention = 0.7  # Base 70% retention

        # Adjust based on video factors
        if local_analysis.duration > 600:  # Longer than 10 minutes
            base_retention *= 0.8

        if viral_factors["content_pacing"] > 0.7:
            base_retention *= 1.2

        if viral_factors["production_quality"] > 0.8:
            base_retention *= 1.1

        return min(base_retention, 1.0)  # Cap at 100%

    def _estimate_potential_engagement(
        self, youtube_metrics, viral_factors: Dict[str, float]
    ) -> float:
        """Estimate potential engagement rate based on benchmark and content quality"""
        benchmark_engagement = youtube_metrics.engagement_rate
        quality_multiplier = sum(viral_factors.values()) / len(viral_factors)

        return benchmark_engagement * quality_multiplier

    def _analyze_target_demographics(
        self, youtube_metrics, local_analysis
    ) -> List[Dict[str, str]]:
        """Analyze and suggest target demographics"""
        # Base analysis on content type and performance
        targets = []

        # Content-based targeting
        if local_analysis.duration < 180:  # Short videos
            targets.append(
                {
                    "age_range": "18-24",
                    "platforms": "YouTube Shorts, TikTok, Instagram",
                    "interests": "Fast-paced content, entertainment",
                    "reason": "Short attention span, mobile-first viewers",
                }
            )

        if youtube_metrics.category_id in ["20", "24"]:  # Gaming/Entertainment
            targets.append(
                {
                    "age_range": "13-34",
                    "platforms": "YouTube, Twitch, Discord",
                    "interests": "Gaming, entertainment, technology",
                    "reason": "Core gaming and entertainment audience",
                }
            )

        if (
            "tutorial" in youtube_metrics.title.lower()
            or "how to" in youtube_metrics.title.lower()
        ):
            targets.append(
                {
                    "age_range": "25-44",
                    "platforms": "YouTube, LinkedIn, Pinterest",
                    "interests": "Education, self-improvement, DIY",
                    "reason": "Professional development and learning focus",
                }
            )

        return targets

    def _generate_hashtags(self, youtube_metrics, local_analysis) -> List[str]:
        """Generate recommended hashtags"""
        hashtags = []

        # Add category-based hashtags
        category_tags = {
            "20": ["gaming", "gameplay", "streamer"],
            "24": ["entertainment", "trending", "viral"],
            "26": ["howto", "tutorial", "learning"],
        }
        hashtags.extend(category_tags.get(youtube_metrics.category_id, []))

        # Add quality-based hashtags
        if local_analysis.resolution[0] >= 1920:
            hashtags.append("HD")

        # Add existing tags as hashtags
        hashtags.extend(
            [tag.lower().replace(" ", "") for tag in youtube_metrics.tags[:3]]
        )

        return list(set(hashtags))  # Remove duplicates

    def _calculate_growth_potential(
        self, viral_factors: Dict[str, float], competition: str
    ) -> str:
        """Calculate growth potential"""
        potential_score = sum(viral_factors.values()) / len(viral_factors)

        if "Low" in competition:
            potential_score *= 1.5
        elif "High" in competition:
            potential_score *= 0.7

        if potential_score > 0.8:
            return "High - Strong viral potential"
        elif potential_score > 0.6:
            return "Medium - Good growth opportunity"
        else:
            return "Low - Consider content optimization"
