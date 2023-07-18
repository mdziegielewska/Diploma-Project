from vidstab import VidStab
import matplotlib.pyplot as plt

def stabilize(input_path, output_path):
    stabilizer = VidStab()
    stabilizer.stabilize(input_path='videos/180_Trim3.mp4', output_path='stable_180_Trim3.avi')

    stabilizer.plot_trajectory()
    plt.show()

    stabilizer.plot_transforms()
    plt.show()