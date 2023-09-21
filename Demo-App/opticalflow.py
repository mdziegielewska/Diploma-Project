import opyf
import utils


def analyze_frames(element):
    dir = "/media/madziegielewska/Seagate Expansion Drive/Diploma-Project/"

    analyzer = opyf.frameSequenceAnalyzer(f"{dir}Demo-App/static/segmentation_results/{element}")
    num = analyzer.number_of_frames

    utils.delete_files_in_directory(f"{dir}Demo-App/static/opyflow_results/{element}")

    analyzer.writeGoodFeaturesPositionsAndDisplacements(fileFormat='csv', outFolder=f"{dir}Demo-App/static/opyflow_results/{element}")
    analyzer.extractGoodFeaturesPositionsDisplacementsAndInterpolate()

    analyzer.writeVelocityField(fileFormat='csv', outFolder=f"{dir}Demo-App/static/opyflow_results/{element}")
    analyzer.set_vecTime(Ntot=num-1,shift=1,step=1)

    analyzer.extractGoodFeaturesAndDisplacements(display='quiver', displayColor=True, saveImgPath=f"{dir}/Demo-App/static/opyflow_results/{element}", width=0.005)

    utils.convert_frames_to_video(f'{element}.mp4', element, 10)