import os
import ffmpeg

# Configuration constants
GRID_ROWS = 7           # Number of rows in the grid
GRID_COLS = 7           # Number of columns in the grid
SCALE_WIDTH = 320       # Width to scale each video thumbnail
SCALE_HEIGHT = 171 // 2 * 2      # Height to scale each video thumbnail, must be even for codec
OUTPUT_MP4 = 'eval_thumbnails.mp4'  # Output MP4 filename
OUTPUT_GIF = 'eval_thumbnails.gif'  # Output GIF filename
FPS = 10                # Frame rate for the GIF to reduce size

# Directory containing the video files (change if necessary)
directory = 'data\\excavator_qpos_oct2\\ckpt_230_8_5_10_256_32_24000_1e-05_42'

# Collect and sort video files that match the pattern "video*-reward-*.mp4"
videos = sorted([f for f in os.listdir(directory) if f.startswith("video") and f.endswith(".mp4")])

# Limit to the first GRID_ROWS * GRID_COLS videos for the grid
num_videos = GRID_ROWS * GRID_COLS
videos = videos[:num_videos]

# Prepare input streams for each video
inputs = [ffmpeg.input(os.path.join(directory, video)) for video in videos]

# Define scale for each video based on specified width and height
scaled_videos = [input.video.filter('scale', SCALE_WIDTH, SCALE_HEIGHT) for input in inputs]

# Build rows by horizontally stacking videos according to GRID_COLS
rows = []
for i in range(0, num_videos, GRID_COLS):
    row = ffmpeg.filter([scaled_videos[j] for j in range(i, i + GRID_COLS)], 'hstack', GRID_COLS)
    rows.append(row)

# Stack the rows vertically to form the full grid according to GRID_ROWS
grid = ffmpeg.filter(rows, 'vstack', GRID_ROWS)

# Split the grid output into two branches

# Output the final video as MP4
output_mp4 = ffmpeg.output(
    grid,
    OUTPUT_MP4,
    vcodec='libx264',  # Use H.264 codec
    video_bitrate='3000k',  # Set bitrate
    crf=18,  # Set constant rate factor
    preset='slow',  # Use slower preset for better compression
    pix_fmt='yuv420p'  # Ensure compatibility with most players
).overwrite_output()
ffmpeg.run(output_mp4)


# Generate a palette for better GIF quality
palette = (
    ffmpeg
    .input(OUTPUT_MP4)
    .filter('palettegen', stats_mode='diff')
    .output('palette.png', frames=1)
    .overwrite_output()
)

# Run the palette generation process
ffmpeg.run(palette)

# Create the looping GIF using the palette
output_gif = (
    ffmpeg
    .input(OUTPUT_MP4)
    .filter('fps', fps=10)  # Adjust the frame rate as needed
    .filter('scale', width=GRID_COLS * SCALE_WIDTH, height=GRID_ROWS * SCALE_HEIGHT)
    .filter('paletteuse', dither='none')  # Disable dithering
    .output(OUTPUT_GIF, loop=0)  # loop=0 makes the GIF loop indefinitely
    .global_args('-i', 'palette.png')
    .overwrite_output()
)

# Run the GIF output process
ffmpeg.run(output_gif)

# Clean up the palette image
os.remove('palette.png')