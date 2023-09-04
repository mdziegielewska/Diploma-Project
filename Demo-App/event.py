import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
import numpy as np
import pandas as pd
from TransNetV2.inference import TransNetV2
from datetime import datetime
import utils


def predict_transnetv2(video_path):
    dir = f"/media/madziegielewska/Seagate Expansion Drive/MAGISTERKA/diploma-project/"
    # read model
    model = TransNetV2()

    # get file name and extensions
    file_path = f"{dir}Demo-App/static/uploads/{video_path}"
    filename, file_extension = os.path.splitext(f'{video_path}')

    output_path_preds = f"{dir}Demo-App/static/predictions/{filename}"
    output_path_sce = f"{dir}Demo-App/static/scenes/{filename}"

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(file_path)

    # save predictions and predicted scenes
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=0.005)
    np.savetxt(output_path_sce + "_scenes.txt", scenes, fmt="%d")

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    np.savetxt(output_path_preds + ".predictions.txt", predictions, fmt="%.6f")

    # get most meaningful scene
    frame_result = np.argmax(single_frame_predictions)
    # convert got frame to timestamp
    timestamp_result = utils.convert_frame_to_timestamp(file_path, frame_result)
    
    return frame_result, timestamp_result


def predict_scenedetect(video_path):
    dir = f"/media/madziegielewska/Seagate Expansion Drive/MAGISTERKA/diploma-project/"

    # get file name and extensions
    file_path = f"{dir}Demo-App/static/uploads/{video_path}"
    filename, file_extension = os.path.splitext(f'{video_path}')

    video = open_video(file_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)

    scene_manager.add_detector(
        ContentDetector(threshold=11.5))
    
    # detect potential boundaries of scenes
    scene_manager.detect_scenes(video, show_progress=False)

    # save processed video statistics
    stats_manager.save_to_csv(f'{dir}Demo-App/static/graphs/scenedetect_metrics.csv')

    graph = pd.read_csv(f'{dir}Demo-App/static/graphs/scenedetect_metrics.csv')
    content_val = graph[graph['content_val'] > 3]

    fig = content_val.plot.line(x='Frame Number', y='content_val', title=f'{filename}').get_figure()
    fig.savefig(f'{dir}Demo-App/static/graphs/{filename}.jpg')

    max_value = graph['content_val'].idxmax()
    change = graph[graph['Frame Number'] == max_value]

    timestamp_result = change.iloc[0]['Timecode']
    dt_obj = datetime.strptime(timestamp_result,"%H:%M:%S.%f")
    time = dt_obj.time()

    # convert timestamp to seconds
    sec = time.second
    msec = time.microsecond

    frame_result = utils.convert_timestamp_to_frame(file_path, timestamp_result)
    timestamp_result = round((sec + msec / 1000000), 2)
    
    return frame_result, timestamp_result
