from moviepy.video.io.VideoFileClip import VideoFileClip


# For some reason its only working on video that has never been trimmer before
def trim_video(input_file, output_file, start_time, end_time):
    video = VideoFileClip(input_file)
    clipped_video = video.subclip(start_time, end_time)
    clipped_video.write_videofile(output_file, codec='libx264')


file_name = input("Enter video name (without .mp4): ")
input_file = "videos/" + file_name + ".mp4"
output_file = "videos/" + file_name + "_t.mp4"
start_time = input("Start in s:")
end_time = input("End in s:")

trim_video(input_file, output_file, start_time, end_time)
