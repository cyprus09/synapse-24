from caption_generator import VideoThumbnailGenerator

try:
    # Initialize the generator
    generator = VideoThumbnailGenerator()
    
    # Process video
    results = generator.process_video('../test-videos/elephant-video.mp4', num_thumbnails=6)
    
    # Save results
    generator.save_thumbnails(results)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")