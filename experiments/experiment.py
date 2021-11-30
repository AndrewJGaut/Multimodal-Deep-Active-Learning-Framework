from test_framework.metrics import *
from test_framework.tester import *

from typing import List
import numpy as np
from tqdm.notebook import trange as trange
import matplotlib.pyplot as plt

from pathlib import Path
import os

import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from utils.data_utils import *
from test_framework.model_interface import ModelInterface
from test_framework.tester import Tester
from test_framework.metrics import *
from active_learning.categorical_query_functions import *
from active_learning.gradient_embedding import compute_gradient_embeddings
from active_learning.cluster_margin import *

import traceback



MAIN_IMG_DIMS = (64,64)
SECONDARY_IMG_DIMS = (32, 32)
MAX_SECONDARY_IMAGES = 5

# Model Constants
FINAL_LAYER_LEN = 64

# Training Constants
TEST_DATA_FRACTION = 0.05

class Experiment:

    def __init__(self, models, query_function_names,
                 test_data_fraction=0.05, final_model_layer_len=64,
                 active_learning_batch_size=256, training_epochs=4, test_repeat_count=2):
        self.models = models
        self.query_function_names = query_function_names
        self.path_to_data = "./data/kaggle_satellite_image_classification"
        self.num_classes = 4
        self.main_image_dims = (64, 64)
        self.secondary_img_dims = (32, 32)
        self.max_secondary_images = 5
        self.test_data_fraction = test_data_fraction
        self.final_model_layer_len = final_model_layer_len

        self.active_learning_batch_size = active_learning_batch_size
        self.training_epochs = training_epochs
        self.test_repeat_count = test_repeat_count

        self.tester = None


    def load_dataset(self, max_samples:int = None):
        """

        :param max_samples:
        :return:
        """
        def transform_to_multimodal(image):
            main_image = transforms.Compose([
                transforms.CenterCrop(MAIN_IMG_DIMS),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(image.copy())

            secondary_images = [
                transforms.Compose([
                    transforms.RandomCrop(SECONDARY_IMG_DIMS),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, contrast_factor=0.8))
                ])(image.copy()) for i in range(MAX_SECONDARY_IMAGES)
                ]
            secondary_images = torch.stack(secondary_images)

            return main_image, secondary_images

        dataset = datasets.ImageFolder(self.path_to_data, transform=transform_to_multimodal)
        output_sample_count = len(dataset) if max_samples is None or max_samples >= len(dataset) else max_samples
        all_data_dataloader = torch.utils.data.DataLoader(dataset, batch_size=output_sample_count, num_workers=0,
                                                          shuffle=True)
        (main_image_all, secondary_images_all), y_all = next(iter(all_data_dataloader))

        # Convert y to one-hot array
        y_all = torch.eye(self.num_classes)[y_all]

        return main_image_all.numpy(), secondary_images_all.numpy(), y_all.numpy()


    def get_plot_name(self, model_name, query_function_name):
        return os.path.join("outputs",
                            f"{model_name}_{query_function_name}.png")

    def plot(self, model_name, active_learning_method):
        outfile_path = self.get_plot_name(model_name, active_learning_method)
        Path('/'.join(outfile_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)  # create directory if necessary
        # plot the file
        self.tester.plot_results(plot_savename=outfile_path)


    def run_experiments(self):
        # load data
        x_main, x_secondary, y = self.load_dataset()

        # configure the tester
        self.tester = Tester([x_main, x_secondary], y)
        self.tester.INITIAL_TRAIN_DATA_FRACTION = self.test_data_fraction
        self.tester.ACTIVE_LEARNING_BATCH_SIZE = self.active_learning_batch_size
        self.tester.TRAINING_EPOCHS = self.training_epochs
        self.tester.TEST_REPEAT_COUNT = self.test_repeat_count

        for model in self.models:
            for query_function_name in self.query_function_names:
                try:
                    curr_model = model(query_function_name)
                    print("working on {}".format(curr_model.name()))
                    self.tester.test_model(curr_model)
                    self.plot(curr_model.name(), query_function_name)
                except Exception as e:
                    print(f"Got exception {e} for model {curr_model.name()} with stack trace:\n{traceback.print_exc()}")

            plt.clear()
