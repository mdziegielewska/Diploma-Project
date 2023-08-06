from vidstab import VidStab
from moviepy.editor import *
from natsort import natsorted
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip


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
    print("converting video from ", video_path)

    # read video
    vidcap = cv2.VideoCapture(video_path)
    ret,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    count = 0
    while ret:
        cv2.imwrite(f"{output_path}/frame%d.png" % count, image) # save frame as png file      
        ret,image = vidcap.read()
        count += 1

    print(f"video converted to {count} frames")


def convert_video_to_array(video_path):
    print("converting video from ", video_path)

    # read video
    vidcap = cv2.VideoCapture(video_path)
    ret,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    count = 0
    frames = []
    while ret:
        image = cv2.resize(image, (384, 384))
        frames.append(image)
        count += 1

    print(f"video converted to {count} frames")
    frames = np.array(frames)

    return frames


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
    print("converting frames from ", frames_path)

    file_list = glob.glob(f'{frames_path}/*.png')  # get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)

    clips = [ImageClip(m).set_duration(0.03)
            for m in file_list_sorted]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_name, fps=fps)


def read_files(path):
    files = []

    file_list = glob.glob(f"{path}/*.*")  # get all the filesin the current directory
    file_list_sorted = natsorted(file_list,reverse=False)

    for file in file_list_sorted:
        files.append(file)

    files = np.array(files)
    len(f"read {len(files)} files")

    return files


def read_image_files(path, H, W, interpolation=False):
    files = []

    file_list = glob.glob(f"{path}/*.*")  # get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)

    if interpolation == False:
        for file in file_list_sorted:
            img = cv2.imread(file, 1)
            img = cv2.resize(img, (H, W))
            files.append(img)
    else:
        for file in file_list_sorted:
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (H, W), interpolation = cv2.INTER_NEAREST)  # otherwise ground truth changes 
            files.append(img)

    files_arr = np.array(files)
    len(f"read {len(files)} images")

    return files_arr


def get_average_pixels(mask_list, return_weights=False):
    background = 0
    needle = 0
    oocyte = 0
    spermatozoid = 0

    for y in mask_list:
        background += y.size - np.count_nonzero(y)
        needle += np.count_nonzero(y == 1)
        oocyte += np.count_nonzero(y == 2)
        spermatozoid += np.count_nonzero(y == 3)

    # get average pixels on mask
    print("Needle: ", round(needle/len(mask_list)))
    print("Ooctye: ", round(oocyte/len(mask_list)))
    print("Spermatozoid: ", round(spermatozoid/len(mask_list)))
    print("Background: ", round(background/len(mask_list)))
    print()

    if return_weights==True:
        return get_weights(background, needle, oocyte, spermatozoid)


def get_weights(b, n, o, s):
    classes_count_mean = [0,0,0,0]

    classes_count_mean[0] = b
    classes_count_mean[1] = n
    classes_count_mean[2] = o
    classes_count_mean[3] = s

    classes_count_mean = [1/x for x in classes_count_mean]
    classess_weights = classes_count_mean / np.linalg.norm(classes_count_mean)

    return classess_weights


def augment_data(images, masks, size_x, size_y):
    augmented_images = []
    augmented_masks = []

    # iterate over list of images and masks
    for x, y in zip(images, masks):
        augmented_images.append(x)
        augmented_masks.append(y)

        aug = CenterCrop(size_x, size_y, p=1.0)
        augmented = aug(image=x, mask=y)
        x1 = augmented['image']
        y1 = augmented['mask']

        augmented_images.append(x1)
        augmented_masks.append(y1)

        aug = RandomRotate90(p=1.0)
        augmented = aug(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        augmented_images.append(x2)
        augmented_masks.append(y2)

        aug =GridDistortion(p=1.0)
        augmented = aug(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']

        augmented_images.append(x3)
        augmented_masks.append(y3)

        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        augmented_images.append(x4)
        augmented_masks.append(y4)

        aug = VerticalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        augmented_images.append(x5)
        augmented_masks.append(y5)

    return augmented_images, augmented_masks


def remove_background(image):
  tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
  b, g, r = cv2.split(image)
  rgba = [b,g,r, alpha]
  dst = cv2.merge(rgba,4)

  return image


def count_frames_manual(video):
	total = 0
        
	while True:
		(grabbed, frame) = video.read()
                
		if not grabbed:
			break
		total += 1

	return total