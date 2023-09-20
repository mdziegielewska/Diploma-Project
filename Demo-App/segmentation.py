# py script to test segmentation, later to be used in a demo app
import os
import cv2
import numpy as np
import glob
import utils
import opticalflow as of
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.models import load_model
from moviepy.editor import *
from natsort import natsorted
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import matplotlib.pyplot as plt

# print(tf.config.list_physical_devices('GPU'))
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

directory = "/media/madziegielewska/Seagate Expansion Drive/Diploma-Project"

model = "unet"
backbone = "resnet50"
model_path = f"{directory}/Demo-App/models/{model}_softmax_1500_{backbone}.hdf5"
output_path = f"{directory}/Demo-App/static/segmentation_results"


def test_segmentation(video_name, element, statistics=True):
    input_path = f"{directory}/videos/{video_name}"

    utils.delete_files_in_directory(output_path)

    # load model and preprocessing
    model = load_model(model_path, compile=False)
    preprocess_input = sm.get_preprocessing(backbone)

    # read video
    vidcap = cv2.VideoCapture(input_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    count = 0
    frames = []
    gray_results = []
    results = []

    while True:
        ret,image = vidcap.read()

        if not ret:
            break

        frame = cv2.resize(image, (384, 384))
        frames.append(frame)

        frame_input = np.expand_dims(frame, 0)
        test_frame = preprocess_input(frame_input)
        test = model.predict(test_frame)
        test_pred = np.argmax(test, axis=3)[0,:,:]

        gray_results.append(test_pred)

        """
        needle_mask = test_pred.astype(np.uint8)
        needle_mask[needle_mask != 1] = 0
        needle_mask[needle_mask == 1] = 255
        needle_mask = np.stack((needle_mask,)*3, axis=-1)
        needle_mask[np.where((needle_mask==[255,255,255]).all(axis=2))] = [240, 100, 50]

        oocyte_mask = test_pred.astype(np.uint8)
        oocyte_mask[oocyte_mask != 2] = 0
        oocyte_mask[oocyte_mask == 2] = 255
        oocyte_mask = np.stack((oocyte_mask,)*3, axis=-1)
        oocyte_mask[np.where((oocyte_mask==[255,255,255]).all(axis=2))] = [0,255,255]

        spermatozoid_mask = test_pred.astype(np.uint8)
        spermatozoid_mask[spermatozoid_mask != 3] = 0
        spermatozoid_mask[spermatozoid_mask == 3] = 255
        spermatozoid_mask = np.stack((spermatozoid_mask,)*3, axis=-1)
        spermatozoid_mask[np.where((spermatozoid_mask==[255,255,255]).all(axis=2))] = [0,0,255]
        """

        #spermatozoid_mask = get_masks(test_pred, "spermatozoid")
        dst = get_masks(test_pred, element)
        #needle_mask = get_masks(test_pred, "needle")
        frame = np.uint8(frame)
        dst = np.uint8(dst)

        #dst = cv2.addWeighted(oocyte_mask, 1, needle_mask, 1, 0)
        #dst = cv2.addWeighted(dst, 1, spermatozoid_mask, 1, 0)
        masked = cv2.bitwise_and(frame,dst)
        combined_masks = utils.remove_background(masked)
        

        # dst = cv2.addWeighted(frame, 1, combined_masks, 0.6, 0)
        cv2.imwrite(f"{output_path}/frame_{count}.png", combined_masks)

        results.append(dst)
        count += 1

    vidcap.release()

    preds = np.array(results)
    of.analyze_frames()

    utils.convert_frames_to_video(video_name, 20)

    if statistics == True:
        stats = utils.get_average_pixels(gray_results)

        return stats


def get_masks(pred, element):
    if element == "spermatozoid":
        mask = [0,0,255]
        label = 3
    elif element == "needle":
        mask = [240, 100, 50]
        label = 1
    elif element == "oocyte":
        mask = [0,255,255]
        label = 2

    element_mask = pred.astype(np.uint8)
    element_mask[element_mask != label] = 0
    element_mask[element_mask == label] = 255
    element_mask = np.stack((element_mask,)*3, axis=-1)
    #element_mask[np.where((element_mask==[255,255,255]).all(axis=2))] = mask

    return element_mask
