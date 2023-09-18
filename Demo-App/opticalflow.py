import opyf
import utils


def analyze_frames():
    dir = "/media/madziegielewska/Seagate Expansion Drive/Diploma-Project/"

    analyzer = opyf.frameSequenceAnalyzer(f"{dir}Demo-App/static/segmentation_results")
    num = analyzer.number_of_frames

    utils.delete_files_in_directory(f"{dir}Demo-App/static/opyflow_results")

    analyzer.writeGoodFeaturesPositionsAndDisplacements(fileFormat='csv', outFolder=f"{dir}Demo-App/static/opyflow_results")
    analyzer.extractGoodFeaturesPositionsDisplacementsAndInterpolate()

    analyzer.writeVelocityField(fileFormat='csv', outFolder=f"{dir}Demo-App/static/opyflow_results")
    analyzer.set_vecTime(Ntot=num//2,shift=2,step=1)

    analyzer.extractGoodFeaturesAndDisplacements(display='quiver', displayColor=True, saveImgPath=f"{dir}/Demo-App/static/opyflow_results", width=0.002)