import igl  # You need to import igl first to avoid some errors
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from dataloader.icosahedron_dataset import ico_to_erp

parser = argparse.ArgumentParser(description="Visualize depth on icosahedron")
parser.add_argument("depth_file", metavar="DEPTH_FILE", help="path to depth npy file")


if __name__ == "__main__":
    args = parser.parse_args()
    # Load depth on icosahedron
    ico_depth = np.load(args.depth_file)
    vertex_num = ico_depth.shape[1]
    level = int(math.log((vertex_num - 2) // 10, 4))

    # Visualize as equirectangular images
    viz_depth = ico_to_erp(ico_depth, level)
    plt.imshow(viz_depth)
    plt.show()
