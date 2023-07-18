from vidstab import VidStab
from moviepy.editor import *
from natsort import natsorted
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def stabilize_video(input_path, output_path):
    print("stabilizing video from ", input_path)

    stabilizer = VidStab()
    # stabilize video
    stabilizer.stabilize(input_path=input_path, output_path=output_path)

    stabilizer.plot_trajectory()
    plt.show()

    stabilizer.plot_transforms()
    plt.show()


def convert_video_to_frames(video_path, output_path):
    base_dir = os.path.realpath(video_path)
    print("converting video from ", base_dir)

    # read video
    vidcap = cv2.VideoCapture(base_dir)
    ret,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    count = 0
    while ret:
        cv2.imwrite(f"{output_path}/frame%d.png" % count, image) # save frame as png file      
        ret,image = vidcap.read()
        count += 1

    print(f"video converted to {count} frames")


def convert_timestamp_to_frame(video_path, timestamp):
    base_dir = os.path.realpath(video_path)

    vidcap = cv2.VideoCapture(base_dir)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    dt_obj = datetime.strptime(timestamp,"%H:%M:%S.%f")
    time = dt_obj.time()

    # convert timestamp to seconds
    min = time.minute
    sec = time.second
    msec = time.microsecond

    milisec = min * 60 + sec + msec / 1000000
    frame_number = round(milisec * fps + 1)

    return frame_number


def convert_frames_to_video(frames_path, video_name, fps):
    base_dir = os.path.realpath(frames_path)
    print("converting frames from ", base_dir)

    file_list = glob.glob(f'{base_dir}/*.png')  # get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)

    clips = [ImageClip(m).set_duration(0.03)
            for m in file_list_sorted]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_name, fps=fps)


def read_files(path):
    base_dir = os.path.realpath(path)
    print(base_dir)

    files = []

    file_list = glob.glob(f"{base_dir}/*.*")  # get all the filesin the current directory
    file_list_sorted = natsorted(file_list,reverse=False)

    for file in file_list_sorted:
        files.append(file)

    files = np.array(files)
    len(f"read {len(files)} files")

    return files


def read_image_files(path, H, W):
    base_dir = os.path.realpath(path)
    print(base_dir)

    files = []

    file_list = glob.glob(f"{base_dir}/*.png")  # get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)

    for file in file_list_sorted:
        img = cv2.imread(file, 1)
        img = cv2.resize(img, (H, W))
        files.append(img)

    files = np.array(files)
    len(f"read {len(files)} images")

    return files


if __name__ == "__main__":
   # function to run
   directory = "/media/madziegielewska/Seagate Expansion Drive/MAGISTERKA/diploma-project"
   #convert_video_to_frames(f"{directory}/videos/21_Trim2.mp4", "frames")

   frame = convert_timestamp_to_frame(f"{directory}/videos/100_Trim1.mp4", "00:00:16.704")
   print(frame)