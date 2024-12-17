from storyboard_generator import ContentStoryboardGenerator
import cv2
import numpy as np
from pathlib import Path
import argparse


def analyze_local_video(video_path):
    """Analyze metrics from a local video file"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps  # in seconds
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate basic metrics
        local_metrics = {
            "duration": duration,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "filesize": Path(video_path).stat().st_size / (1024 * 1024),  # in MB
        }

        # Sample frames for content analysis
        frame_samples = []
        sample_intervals = np.linspace(0, total_frames - 1, 10, dtype=int)

        for frame_no in sample_intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                # Calculate average brightness and movement (simplified)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_samples.append(
                    {"brightness": np.mean(gray), "frame_no": frame_no}
                )

        cap.release()

        # Add content analysis metrics
        local_metrics.update(
            {
                "avg_brightness": np.mean(
                    [sample["brightness"] for sample in frame_samples]
                ),
                "frame_samples": frame_samples,
            }
        )

        return local_metrics

    except Exception as e:
        raise Exception(f"Error analyzing local video: {str(e)}")


def compare_videos(local_metrics, youtube_metrics, storyboard):
    """Compare local video metrics with YouTube video metrics"""
    # Estimate YouTube video duration based on engagement points
    youtube_duration = 0
    if storyboard.engagement_points:
        last_point = storyboard.engagement_points[-1]
        minutes, _ = map(int, last_point["timing"].split(":"))
        youtube_duration = minutes * 60  # Convert to seconds

    comparison = {
        "technical_comparison": {
            "duration_difference": abs(local_metrics["duration"] - youtube_duration),
            "resolution": local_metrics["resolution"],
            "fps": local_metrics["fps"],
            "filesize": f"{local_metrics['filesize']:.2f}MB",
        },
        "content_analysis": {
            "brightness_score": local_metrics["avg_brightness"],
            "youtube_engagement": youtube_metrics.engagement_rate,
            "youtube_retention": youtube_metrics.viewer_retention,
        },
        "recommendations": [],
    }

    # Generate recommendations based on comparison
    if local_metrics["duration"] > youtube_duration * 1.2:
        comparison["recommendations"].append(
            "Consider shortening your video - successful reference is shorter"
        )
    elif local_metrics["duration"] < youtube_duration * 0.8:
        comparison["recommendations"].append(
            "Consider adding more content - successful reference is longer"
        )

    # Add storyboard-based recommendations
    for point in storyboard.engagement_points:
        time_sec = int(point["timing"].split(":")[0]) * 60
        if time_sec < local_metrics["duration"]:
            comparison["recommendations"].append(
                f"At {point['timing']}: {point['suggestion']}"
            )

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare local video with YouTube video"
    )
    parser.add_argument("--local-video", required=True, help="Path to local video file")
    parser.add_argument(
        "--youtube-id", required=True, help="YouTube video ID to compare against"
    )
    parser.add_argument("--api-key", required=True, help="YouTube API key")

    args = parser.parse_args()

    try:
        # Initialize the generator
        generator = ContentStoryboardGenerator(args.api_key)

        # Analyze local video
        print(f"Analyzing local video: {args.local_video}")
        local_metrics = analyze_local_video(args.local_video)
        print("Local video analysis completed")

        # Analyze YouTube video
        print(f"Analyzing YouTube video: {args.youtube_id}")
        youtube_metrics = generator.analyze_video(args.youtube_id)
        print("YouTube video analysis completed")

        # Generate storyboard based on successful YouTube video
        storyboard = generator.generate_storyboard(
            video_metrics=youtube_metrics,
            target_platform="youtube",
            content_type="vlog",
        )

        # Compare videos and generate recommendations
        comparison = compare_videos(local_metrics, youtube_metrics, storyboard)

        # Print results
        print("\nComparison Results:")
        print("==================")

        print("\nTechnical Comparison:")
        for key, value in comparison["technical_comparison"].items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("\nContent Analysis:")
        for key, value in comparison["content_analysis"].items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("\nRecommendations:")
        for idx, rec in enumerate(comparison["recommendations"], 1):
            print(f"{idx}. {rec}")

        # Print predicted performance
        performance = generator.predict_performance(storyboard)
        print("\nPredicted Performance (based on YouTube reference):")
        print(f"Engagement Rate Target: {performance['predicted_engagement']:.2%}")
        print(f"Retention Rate Target: {performance['estimated_retention']:.2%}")
        print(f"Viral Potential: {performance['viral_potential']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return


if __name__ == "__main__":
    main()
