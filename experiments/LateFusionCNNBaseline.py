
from models.multimodal.late_fusion_model import LateFusionModel
from models.unimodal.squeezenet import SqueezeNet
from models.multimodal.combination_functions import MEAN_CLASSIFICATION
from test_framework.tester import Tester
import numpy as np
from utils.data_utils import get_kaggle_satellite_image_classification_dataset_as_numpy_arrays
import active_learning.categorical_query_functions as query_functions



PATH_TO_DATA = "./data/kaggle_satellite_image_classification"
active_learning_functions = [query_functions.RANDOM, query_functions.MIN_MAX, query_functions.MAX_ENTROPY, query_functions.MIN_MARGIN]
active_learning_function_descriptions = ["Random", "Min-Max", "Max-Ent", "Min-Margin"]

if __name__ == "__main__":
    # get data
    first_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA)
    second_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA, True)
    tester_x = np.stack((first_modality[0], second_modality[0]), axis=1) #np.stack((first_modality[0], second_modality[0]))
    tester_y = first_modality[1]

    # define tester
    tester = Tester(tester_x, tester_y, training_epochs=3, active_learning_loop_count=16)
    tester.INITIAL_TRAIN_DATA_FRACTION = 0.05

    for i,active_learning_function in enumerate(active_learning_functions):
        # define model
        multimodal_model = LateFusionModel([SqueezeNet(), SqueezeNet()], MEAN_CLASSIFICATION, active_learning_function, name="Multimodal squeezenets model",
                                           details=active_learning_function_descriptions[i])

        # test model
        tester.test_model(multimodal_model)

    # get results
    tester.plot_results()
