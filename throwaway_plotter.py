from test_framework.tester import *
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import pickle
import sklearn.metrics as skm
import os
import pdb


ACTIVE_LEARNING_LOOP_COUNT = 10

'''
    Loads model results list from a given file
    '''


def load_results(savename="test_results.data"):
    with open(savename, "rb") as fp:
        return pickle.load(fp)


'''
Plots and saves train/test curves for all models, along with their ALC measures
'''


def plot_results(model_results, plot_savename="test_results.png", show=False) -> None:
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    #plt.title(plot_savename.split(".")[0])
    fig.suptitle(plot_savename.split(".")[0])

    model_names = [r.model_name for r in model_results]

    # Plot training curve
    for r in model_results:
        x = np.mean(r.data_size, axis=0)
        y = np.mean(r.training_performance, axis=0)
        axarr[0][0].plot(x, y, label=r.model_name)
    axarr[0][0].legend()
    axarr[0][0].set_xlabel("Number of Labelled Training Examples")
    axarr[0][0].set_ylabel("Training Accuracy")
    axarr[0][0].title.set_text("Training Accuracy")

    # Plot testing curve
    for r in model_results:
        x = np.mean(r.data_size, axis=0)
        y = np.mean(r.test_performance, axis=0)
        axarr[1][0].plot(x, y, label=r.model_name)
    axarr[1][0].legend()
    #pdb.set_trace()
    axarr[1][0].set_xlabel("Number of Labelled Training Examples")
    axarr[1][0].set_ylabel("Testing Accuracy")
    axarr[1][0].title.set_text("Testing Accuracy")

    # Calculate ALC measure for each model
    train_alc = []
    for r in model_results:
        y = np.mean(r.training_performance, axis=0)
        train_alc.append(skm.auc(range(ACTIVE_LEARNING_LOOP_COUNT), y))
    test_alc = []
    for r in model_results:
        y = np.mean(r.test_performance, axis=0)
        test_alc.append(skm.auc(range(ACTIVE_LEARNING_LOOP_COUNT), y))

    # Choose bar graph bounds
    min_alc = min(*train_alc, *test_alc)
    max_alc = max(*train_alc, *test_alc)
    bar_ylim_lower = min_alc - 0.1 * (max_alc - min_alc)
    bar_ylim_upper = max_alc + 0.1 * (max_alc - min_alc)

    # Plot train ALC measure for each model
    axarr[0][1].bar(model_names, train_alc, color=[handle.get_color() for handle in axarr[0][0].get_legend().legendHandles])
    axarr[0][1].set_ylim(bar_ylim_lower, bar_ylim_upper)
    axarr[0][1].set_xticklabels(
        model_names,
        fontdict={
            'rotation': 45,
            'horizontalalignment': 'right'
        }
    )
    axarr[0][1].title.set_text("Training ALC")
    axarr[0][1].set_xlabel("AL Method")
    axarr[0][1].set_ylabel("ALC")

    # Plot test ALC measure for each model

    axarr[1][1].bar(model_names, test_alc, color=[handle.get_color() for handle in axarr[1][0].get_legend().legendHandles])
    axarr[1][1].set_ylim(bar_ylim_lower, bar_ylim_upper)
    axarr[1][1].set_xticklabels(
        model_names,
        fontdict={
            'rotation': 45,
            'horizontalalignment': 'right'
        }
    )
    axarr[1][1].title.set_text("Testing ALC")
    axarr[1][1].set_xlabel("AL Method")
    axarr[1][1].set_ylabel("ALC")

    # Save plots
    #plt.title("HELLO")
    plt.savefig(os.path.join("outputs", "FINAL_OUTPUTS", plot_savename), facecolor='white', transparent=False)
    if show:
        plt.show()


'''
Simple struct for Tester to store evaluation info associated with a particular model
'''


class ModelResults:
    def __init__(self, model: ModelInterface, test_count: int, active_learning_loop_count: int):
        # Store name and details to identify model and results if model is lost
        self.model_name = model.name()
        self.model_details = model.details()

        # Store data size after each active learning loop
        self.data_size = np.zeros((test_count, active_learning_loop_count))

        # Store performance on training and test data after each active learning loop
        self.training_performance = np.zeros((test_count, active_learning_loop_count))
        self.test_performance = np.zeros((test_count, active_learning_loop_count))

        # Store training time during each active learning loop
        self.training_time = np.zeros((test_count, active_learning_loop_count))

        # Store querying time during each active learning loop
        self.querying_time = np.zeros((test_count, active_learning_loop_count))

    def add(self, test_iteration, active_learning_iteration, data_size, training_performance, test_performance,
            training_time, querying_time):
        self.data_size[test_iteration, active_learning_iteration] = data_size
        self.training_performance[test_iteration, active_learning_iteration] = training_performance
        self.test_performance[test_iteration, active_learning_iteration] = test_performance
        self.training_time[test_iteration, active_learning_iteration] = training_time
        self.querying_time[test_iteration, active_learning_iteration] = querying_time

if __name__ == '__main__':
    title_name_file_name_pairs = [
        ("linear-model-based-on-sofmax-regression_0.05,64,32,20,8.data", "Linear Model on Regular Dataset"),
        ("linear-model-based-on-sofmax-regression_0.05,64,32,20,8_{GRAYSCALE}.data", "Linear Model on Grayscale Dataset"),
        ("Multimodal-late-fusion-model-based-on-AlexNet_0.05,64,32,20,8.data", "Late Fusion Model on Regular Dataset"),
        ("Multimodal-late-fusion-model-based-on-AlexNet_0.05,64,32,20,8_{GRAYSCALE}.data",
         "Late Fusion Model on Grayscale Dataset"),
        ("Multimodal-middle-fusion-model-based-on-AlexNet_0.05,64,32,20,8.data",
         "Middle Fusion Model on Regular Dataset"),
        ("Multimodal-middle-fusion-model-based-on-AlexNet_0.05,64,32,20,8_{GRAYSCALE}.data",
         "Middle Fusion Model on Grayscale Dataset")
    ]
    #model_results = load_results("outputs/FINAL_OUTPUTS/linear-model-based-on-sofmax-regression_0.05,64,32,20,8.data")
    #plot_results(model_results, "TEST_RESULTS.png")

    for pair in title_name_file_name_pairs:
        model_results = load_results(os.path.join("outputs", "FINAL_OUTPUTS", pair[0]))
        plot_results(model_results, pair[1] + ".png")


