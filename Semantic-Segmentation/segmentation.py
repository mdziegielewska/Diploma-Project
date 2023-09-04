# py script to test segmentation, later to be used in a demo app
import os
import cv2
import numpy as np
import glob
import utils
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


def test_segmentation(video_input_path, output_path, video_name, model_name, model_path, backbone, statistics=True):
    # load model and preprocessing
    model = load_model(model_path, compile=False)
    preprocess_input = sm.get_preprocessing(backbone)

    # read video
    vidcap = cv2.VideoCapture(video_input_path)
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

        dst = cv2.addWeighted(oocyte_mask, 1, needle_mask, 1, 0)
        dst = cv2.addWeighted(dst, 1, spermatozoid_mask, 1, 0)
        combined_masks = utils.remove_background(dst)

        dst = cv2.addWeighted(frame, 1, combined_masks, 0.6, 0)
        cv2.imwrite(f"{output_path}/frame_{count}.png", dst)

        results.append(dst)
        count += 1

    vidcap.release()

    print(f"video converted to {count} frames")

    preds = np.array(results)

    if statistics == True:
        n = np.random.randint(len(preds)-1)

        stats = utils.get_average_pixels(gray_results)

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
        
    utils.convert_frames_to_video(output_path, f"{video_name}_{model_name}_segmentation.mp4", 20)


if __name__ == "__main__":
    directory = "/media/madziegielewska/Seagate Expansion Drive/MAGISTERKA/diploma-project"
    
    video_name = "158_Trim5.mp4"
    input_path = f"{directory}/videos/{video_name}"
    output_path = f"{directory}/Semantic-Segmentation/segmentation_test_results"

    model = "unet"
    model_path = f"{directory}/Semantic-Segmentation/models/{model}_softmax_1500_resnet50.hdf5"

    test_segmentation(input_path, output_path, video_name, model, model_path, "resnet50")