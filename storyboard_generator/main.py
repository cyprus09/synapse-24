from storyboard_generator_generic import StoryBoardGeneratorGeneric, PredictiveMetrics
import cv2
import numpy as np
from pathlib import Path
import argparse
from datetime import timedelta


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_video_metrics(metrics, is_youtube=True):
    source = "YouTube" if is_youtube else "Local"
    print(f"\n=== {source} Video Analysis ===")

    if is_youtube:
        print(f"\n📊 Engagement Metrics:")
        print(f"  • Views: {metrics.views:,}")
        print(f"  • Likes: {metrics.likes:,}")
        print(f"  • Comments: {metrics.comments:,}")
        print(f"  • Engagement Rate: {metrics.engagement_rate:.2f}%")
        print(f"  • Duration: {format_duration(metrics.duration)}")

        print(f"\n🏷️ Video Details:")
        print(f"  • Title: {metrics.title}")
        print(f"  • Category ID: {metrics.category_id}")
        print(f"  • Published: {metrics.publish_date}")

        if metrics.tags:
            print(f"\n🔖 Tags:")
            print(
                f"  • {', '.join(metrics.tags[:5])}{'...' if len(metrics.tags) > 5 else ''}"
            )
    else:
        print(f"\n📹 Video Properties:")
        print(f"  • Duration: {format_duration(metrics.duration)}")
        print(f"  • Resolution: {metrics.resolution[0]}x{metrics.resolution[1]}")
        print(f"  • FPS: {metrics.fps:.2f}")

        print(f"\n💡 Brightness Analysis:")
        print(f"  • Mean: {metrics.brightness_analysis['mean']:.2f}")
        print(f"  • Std Dev: {metrics.brightness_analysis['std']:.2f}")

        if metrics.audio_analysis:
            print(f"\n🔊 Audio Analysis:")
            print(f"  • Average Volume: {metrics.audio_analysis.average_volume:.2f}")
            print(
                f"  • Background Noise Level: {metrics.audio_analysis.background_noise:.2f}"
            )
            print(f"  • Tempo: {metrics.audio_analysis.tempo:.2f} BPM")

        print(f"\n🎬 Scene Analysis:")
        print(f"  • Scene Changes: {len(metrics.scene_changes)}")
        avg_scene_duration = metrics.duration / (len(metrics.scene_changes) + 1)
        print(f"  • Average Scene Duration: {format_duration(avg_scene_duration)}")


def print_recommendations(recommendations):
    print("\n=== 📋 Recommendations ===")

    if recommendations.title_suggestions:
        print("\n📝 Title Suggestions:")
        for suggestion in recommendations.title_suggestions:
            print(f"  • {suggestion}")

    if recommendations.thumbnail_improvements:
        print("\n🖼️ Thumbnail Improvements:")
        for improvement in recommendations.thumbnail_improvements:
            print(f"  • {improvement}")

    if recommendations.content_improvements:
        print("\n🎥 Content Improvements:")
        for improvement in recommendations.content_improvements:
            print(f"\n  {improvement['area']}:")
            print(f"    • Issue: {improvement['suggestion']}")
            print(f"    • Action: {improvement['action']}")

    if recommendations.optimization_suggestions:
        print("\n⚙️ Technical Optimizations:")
        for suggestion in recommendations.optimization_suggestions:
            print(f"  • {suggestion}")


def print_predictive_metrics(metrics: PredictiveMetrics):
    print("\n=== 🔮 Predictive Analytics ===")

    print("\n📈 Viral Potential:")
    print(f"  • Viral Probability: {metrics.viral_probability:.1f}%")
    print(f"  • 30-Day View Estimate: {metrics.estimated_views_30d:,}")
    print(f"  • Estimated Engagement Rate: {metrics.estimated_engagement_rate:.1f}%")
    print(f"  • Viewer Retention Estimate: {metrics.viewer_retention_estimate:.1%}")

    print("\n⏰ Best Posting Times:")
    for t in metrics.best_posting_times:
        print(f"  • {t.strftime('%I:%M %p')}")

    print("\n👥 Target Demographics:")
    for demo in metrics.target_demographics:
        print(f"\n  Age Range: {demo['age_range']}")
        print(f"  Platforms: {demo['platforms']}")
        print(f"  Interests: {demo['interests']}")
        print(f"  Reasoning: {demo['reason']}")

    print("\n🎯 Content Performance Factors:")
    for factor, score in metrics.content_virality_factors.items():
        print(f"  • {factor.replace('_', ' ').title()}: {score:.2f}")

    print("\n🏷️ Recommended Hashtags:")
    print(f"  • {', '.join(['#' + tag for tag in metrics.recommended_hashtags])}")

    print("\n📊 Market Analysis:")
    print(f"  • Competition Level: {metrics.competition_level}")
    print(f"  • Growth Potential: {metrics.growth_potential}")


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
        # initialize the analyzer
        analyzer = StoryBoardGeneratorGeneric(args.api_key)

        # analyze local video
        print("⌛ Analyzing local video...")
        local_analysis = analyzer.analyze_local_video(args.local_video)
        print("✅ Local video analysis completed")
        print_video_metrics(local_analysis, is_youtube=False)

        # analyze YouTube video
        print(f"\n⌛ Analyzing YouTube video: {args.youtube_id}")
        youtube_metrics = analyzer.analyze_youtube_video(args.youtube_id)
        print("✅ YouTube video analysis completed")
        print_video_metrics(youtube_metrics, is_youtube=True)
        
        # generate predictions
        print("\n⌛ Generating predictive analytics...")
        predictive_metrics = analyzer.predict_performance(youtube_metrics, local_analysis)
        print_predictive_metrics(predictive_metrics)

        # generate and display recommendations
        print("\n⌛ Generating recommendations...")
        recommendations = analyzer.compare_videos(youtube_metrics, local_analysis)
        print_recommendations(recommendations)
        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return


if __name__ == "__main__":
    main()
