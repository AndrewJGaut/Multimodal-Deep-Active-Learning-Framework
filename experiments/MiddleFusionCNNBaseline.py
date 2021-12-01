import sys
sys.path.append("..")
from models.multimodal.middle_fusion_model import MiddleFusionModel
# from models.unimodal.squeezenet import SqueezeNet
# from models.multimodal.combination_functions import MEAN_CLASSIFICATION
from test_framework.tester import Tester
from test_framework.metrics import LABEL_BALANCED_ACCURACY
import numpy as np
from utils.data_utils import get_kaggle_satellite_image_classification_dataset_as_numpy_arrays
from active_learning import categorical_query_functions as query_functions
import collections
import torch

PATH_TO_DATA = "../data/kaggle_satellite_image_classification"
active_learning_functions = [query_functions.RANDOM, query_functions.MIN_MAX, query_functions.MAX_ENTROPY, query_functions.MIN_MARGIN]
active_learning_function_descriptions = ["Random", "Min-Max", "Max-Ent", "Min-Margin"]
use_smaller_dataset = False

if __name__ == "__main__":
    # get data
    first_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA)
    # second_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA, True)
    second_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA, False, True)
    tester_x1 = first_modality[0]
    tester_x2 = second_modality[0]
    x2_image_counts = np.random.randint(0,6,size=(tester_x1.shape[0]))
    tester_y = first_modality[1]

    if use_smaller_dataset:
        order = np.random.permutation(len(tester_y))
        tester_x1 = tester_x1[order]
        tester_x2 = tester_x2[order]
        x2_image_counts = x2_image_counts[order]
        tester_y = tester_y[order]
        tester_x1 = tester_x1[:600]
        tester_x2 = tester_x2[:600]
        x2_image_counts = x2_image_counts[:600]
        tester_y = tester_y[:600]

    tester_y_onehot = np.zeros((tester_y.size, 4))
    tester_y_onehot[np.arange(tester_y.size),tester_y] = 1
    print("distribution of samples over the entire dataset:",collections.Counter(tester_y))
    # print(collections.Counter(first_modality[1]))

    # define tester
    tester = Tester(tester_x1, tester_x2, x2_image_counts, tester_y_onehot, training_epochs=5, active_learning_loop_count=20)
    tester.INITIAL_TRAIN_DATA_FRACTION = 0.01
    tester.ACTIVE_LEARNING_BATCH_SIZE = 1
    tester.METRIC_FUNCTION = LABEL_BALANCED_ACCURACY

    for i,active_learning_function in enumerate(active_learning_functions):
        print(f"Active learning function: {active_learning_function_descriptions[i]}")
        
        # define model
        multimodal_model = MiddleFusionModel(active_learning_function=active_learning_function,
                                            name=active_learning_function_descriptions[i],
                                            random_seed=0)

        # test model
        tester.test_model(multimodal_model)

    # get results
    print("")
    print([mr.__dict__ for mr in tester.model_results])
    tester.plot_results()
