# imports

import os
import cv2
import numpy as np
import glob
import utils
import tensorflow as tf
from keras.models import load_model
from moviepy.editor import *
from natsort import natsorted
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def test_segmentation(input_path, output_path, model, backbone, statistics=True):
    frames = utils.read_image_files(input_path, 384, 384)

    model = load_model(model, compile=False)
    preprocess_input = sm.get_preprocessing(backbone)

    results = []

    for frame in frames:
        frame_input = np.expand_dims(frame, 0)
        test_frame = preprocess_input(frame_input)
        test = model.predict(test_frame)
        test_pred = np.argmax(test, axis=3)[0,:,:]

        results.append(test_pred)

    preds = np.array(results)

    if statistics == True:
        n = np.random.randint(len(preds)-1)

        stats = utils.get_average_pixels(preds)

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Original frame')
        plt.imshow(frames[n], vmin=0, vmax=3, cmap='gray')
        plt.subplot(232)
        plt.title('Prediction mask')
        plt.imshow(preds[n], vmin=0, vmax=3, cmap='jet')
        plt.subplot(233)
        plt.title('Prediction mask on image')
        plt.imshow(frames[n], vmin=0, vmax=3, cmap='gray')
        plt.imshow(preds[n], vmin=0, vmax=3, cmap="jet", alpha=0.3)
        plt.show()


    count = 0

    for i,m in zip(frames, preds):
        needle_mask = m.astype(np.uint8)
        needle_mask[needle_mask != 1] = 0
        needle_mask[needle_mask == 1] = 255
        needle_mask = np.stack((needle_mask,)*3, axis=-1)

        oocyte_mask = m.astype(np.uint8)
        oocyte_mask[oocyte_mask != 2] = 0
        oocyte_mask[oocyte_mask == 2] = 255
        oocyte_mask = np.stack((oocyte_mask,)*3, axis=-1)

        spermatozoid_mask = m.astype(np.uint8)
        spermatozoid_mask[spermatozoid_mask != 3] = 0
        spermatozoid_mask[spermatozoid_mask == 3] = 255
        spermatozoid_mask = np.stack((spermatozoid_mask,)*3, axis=-1)

        dst = cv2.addWeighted(i, 0.5, oocyte_mask, 0.5, 0)
        dst = cv2.addWeighted(dst, 0.5, needle_mask, 0.5, 0)
        dst = cv2.addWeighted(dst, 0.5, spermatozoid_mask, 0.5, 0)

        cv2.imwrite(f"{output_path}/frame_{count}.png", dst)
        count += 1

    utils.convert_frames_to_video(output_path, "test_unet.mp4", 20)


if __name__ == "__main__":
    directory = "/media/madziegielewska/Seagate Expansion Drive/MAGISTERKA/diploma-project/Semantic-Segmentation"
    
    input_path = f"{directory}/frames_to_test_segmentation"
    output_path = f"{directory}/segmentation_test_results"

    model = f"{directory}/models/unet_softmax_1500_resnet50.hdf5"

    test_segmentation(input_path, output_path, model, "resnet50")