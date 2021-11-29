import sys
sys.path.append("..")
from models.multimodal.middle_fusion_model import MiddleFusionModel
# from models.unimodal.squeezenet import SqueezeNet
# from models.multimodal.combination_functions import MEAN_CLASSIFICATION
from test_framework.tester import Tester
import numpy as np
from utils.data_utils import get_kaggle_satellite_image_classification_dataset_as_numpy_arrays
from active_learning import categorical_query_functions as query_functions

PATH_TO_DATA = "../data/kaggle_satellite_image_classification"
active_learning_functions = [query_functions.RANDOM, query_functions.MIN_MAX, query_functions.MAX_ENTROPY, query_functions.MIN_MARGIN]
active_learning_function_descriptions = ["Random", "Min-Max", "Max-Ent", "Min-Margin"]
use_smaller_dataset = False

if __name__ == "__main__":
    # get data
    first_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA)
    second_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA, True)
    tester_x = np.stack((first_modality[0], second_modality[0]), axis=1)
    tester_y = first_modality[1]

    if use_smaller_dataset:
        order = np.random.permutation(len(tester_y))
        tester_x = tester_x[order]
        tester_y = tester_y[order]
        tester_x = tester_x[:600]
        tester_y = tester_y[:600]

    tester_y_onehot = np.zeros((tester_y.size, 4))
    tester_y_onehot[np.arange(tester_y.size),tester_y] = 1
    import collections
    print(collections.Counter(tester_y))
    print(collections.Counter(first_modality[1]))

    # define tester
    tester = Tester(tester_x, tester_y_onehot, training_epochs=10, active_learning_loop_count=10)
    tester.INITIAL_TRAIN_DATA_FRACTION = 0.05

    for i,active_learning_function in enumerate(active_learning_functions):
        # define model
        multimodal_model = MiddleFusionModel(active_learning_function=active_learning_function)

        # test model
        tester.test_model(multimodal_model)

    # get results
    tester.plot_results()
