import imageio
import glob
import os


def create_gif():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Collect all frame files in order
    filenames = sorted(glob.glob(os.path.join(script_dir, 'year_*.png')))

    if not filenames:
        print('No frames found. Make sure you have run the simulation and have year_*.png files in this directory.')
        exit(1)

    with imageio.get_writer(os.path.join(script_dir, 'simulation.gif'), mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print('GIF saved as simulation.gif')