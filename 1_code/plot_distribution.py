import json

from matplotlib import pyplot as plt
import numpy as np

def plot_distribution(all_y, all_pred, target, plot_file):
    # plot surrogate vs gt
    # plt.scatter(all_y, all_pred, s=5)
    bins = 100
    plot_area = np.zeros((bins, bins))
    for y, pred in zip(all_y, all_pred):
        # clip to make sure it's within the plot area
        y_int = np.clip(int(y * bins), 0, bins - 1)
        pred_int = np.clip(int(pred * bins), 0, bins - 1)
        plot_area[pred_int][y_int] += 1

    # plot log scale
    #plot_area = np.where(plot_area != 0, np.log(plot_area), np.zeros_like(plot_area))
    plot_area = np.clip(plot_area, 0, 800)
    plot_area = np.where(plot_area != 0, np.log(plot_area), np.zeros_like(plot_area))

    print(plot_area)

    plt.imshow(plot_area, extent=[0, 1, 1, 0])
    plt.gca().invert_yaxis()
    plt.xlabel(f"Ground Truth {target}")
    plt.ylabel(f"Surrogate Model Prediction")

    plt.tight_layout()

    plt.savefig(f"{plot_file}.png", dpi=300, format="png")
    plt.close()
