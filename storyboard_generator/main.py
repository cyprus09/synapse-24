from storyboard_generator_generic import StoryBoardGeneratorGeneric, PredictiveMetrics, ContentRecommendations
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


def print_recommendations(recommendations: ContentRecommendations):
    print("\n=== 📈 Improvement Recommendations ===")

    print("\n🎯 Key Improvements Needed:")
    for imp in recommendations.key_improvements:
        print(f"\n  {imp['area']} ({imp['impact']} Impact)")
        print(f"  • Issue: {imp['issue']}")
        print(f"  • Solution: {imp['recommendation']}")

    print("\n📝 Title Optimization:")
    for suggestion in recommendations.title_suggestions:
        print(f"  • {suggestion}")

    print("\n🖼️ Thumbnail Improvements:")
    for improvement in recommendations.thumbnail_improvements:
        print(f"  • {improvement}")

    print("\n⚙️ Technical Optimizations:")
    for opt in recommendations.technical_optimizations:
        print(f"\n  {opt['aspect']}")
        print(f"  • Current: {opt['current']}")
        print(f"  • Recommended: {opt['recommended']}")
        print(f"  • Why: {opt['reason']}")

    print("\n🤝 Engagement Strategies:")
    for strategy in recommendations.engagement_strategies:
        print(f"\n  {strategy['aspect']} ({strategy['priority']} Priority)")
        print(f"  • Strategy: {strategy['strategy']}")
        print(f"  • How: {strategy['implementation']}")

    print("\n📚 Lessons from Benchmark:")
    for learning in recommendations.benchmark_learnings:
        print(f"\n  {learning['element']}")
        print(f"  • Observation: {learning['observation']}")
        print(f"  • Application: {learning['application']}")


def print_predictive_metrics(metrics: PredictiveMetrics):
    print("\n=== 🎯 Local Video Potential Analysis ===")

    print("\n📊 Benchmark Comparison:")
    for factor, score in metrics.benchmark_comparison.items():
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"  {status} {factor.replace('_', ' ').title()}: {score:.0%}")

    print("\n🚀 Growth Predictions:")
    print(f"  • Viral Potential: {metrics.viral_probability:.1f}%")
    print(f"  • Expected Views (30 days): {metrics.estimated_views_30d:,}")
    print(f"  • Projected Engagement Rate: {metrics.estimated_engagement_rate:.1f}%")
    print(f"  • Estimated Viewer Retention: {metrics.viewer_retention_estimate:.0%}")

    print("\n💪 Competitive Advantages:")
    for aspect, advantage in metrics.competition_level.items():
        print(f"  • {aspect.title()}: {advantage}")

    print("\n👥 Target Audience:")
    for demo in metrics.target_demographics:
        print(f"\n  Segment: {demo['age_range']}")
        print(f"  • Platforms: {demo['platforms']}")
        print(f"  • Interests: {demo['interests']}")
        print(f"  • Strategy: {demo['reason']}")

    print("\n⏰ Recommended Posting Times:")
    for t in metrics.best_posting_times:
        print(f"  • {t.strftime('%I:%M %p')}")

    print("\n🔑 Success Factors:")
    for factor, score in metrics.content_virality_factors.items():
        quality = (
            "Excellent"
            if score > 0.8
            else "Good" if score > 0.6 else "Needs Improvement"
        )
        print(f"  • {factor.replace('_', ' ').title()}: {quality} ({score:.0%})")

    print("\n#️⃣ Recommended Hashtags:")
    print(f"  • {', '.join(['#' + tag for tag in metrics.recommended_hashtags])}")

    print(f"\n📈 Overall Growth Potential: {metrics.growth_potential}")


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
        predictive_metrics = analyzer.predict_performance(
            youtube_metrics, local_analysis
        )
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
