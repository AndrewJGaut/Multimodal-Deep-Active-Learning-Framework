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


def get_experiment_configs(initial_train_data_fractions, active_learning_batch_sizes, training_epochs, test_repeat_counts):
    experiment_configs = list()
    for i in range(len(initial_train_data_fractions)):
        experiment_configs.append(ExperimentConfig(initial_train_data_fraction = initial_train_data_fractions[i],
                                                   active_learning_batch_size=active_learning_batch_sizes[i],
                                                   training_epochs=training_epochs[i],
                                                   test_repeat_count=test_repeat_counts[i]))

    return experiment_configs


class ExperimentConfig:
    def __init__(self, initial_train_data_fraction=0.05, final_model_layer_len=64,
                 active_learning_batch_size=256, training_epochs=4, test_repeat_count=2):
        self.initial_train_data_fraction = initial_train_data_fraction
        self.final_model_layer_len = final_model_layer_len

        self.active_learning_batch_size = active_learning_batch_size
        self.training_epochs = training_epochs
        self.test_repeat_count = test_repeat_count

    def __str__(self):
        params = [self.initial_train_data_fraction, self.final_model_layer_len, self.active_learning_batch_size, self.training_epochs, self.test_repeat_count]
        return ",".join([str(x) for x in params])

class Experiment:

    def __init__(self, name, models, query_function_names,
                 query_function_name_to_extra_options=dict(),
                 experiment_configs = list(), grayscale=False,
                 is_test=False):
        """
        :param name
        :param models:
        :param query_function_names:
        :param query_function_name_to_extra_options:
        :param experiment_configs:
        :param grayscale (bool): if True, then grayscale all the images for this experiment so that networks can no
        longer use color to differentiate.
        :param is_test (bool): true if we want to load only part of the data to test the framework.
        """
        self.name = name
        self.models = models
        self.query_function_names = query_function_names
        self.query_function_name_to_extra_options = query_function_name_to_extra_options
        self.path_to_data = "./data/kaggle_satellite_image_classification"
        self.num_classes = 4
        self.main_image_dims = (64, 64)
        self.secondary_img_dims = (32, 32)
        self.max_secondary_images = 5

        self.experiment_configs = experiment_configs

        self.grayscale = grayscale
        self.is_test = is_test

        self.tester = None


    def load_dataset(self, max_samples:int = None):
        """

        :param max_samples:
        :param grayscale: if true, then grayscale all the images. We'll do this in order to see how color affects
        the model (making images grayscale might make the model's job harder, which could increase the utility
        of some of the active learning methods perhaps).
        :return:
        """
        def transform_to_multimodal(image):
            main_image_transforms = [
                transforms.CenterCrop(self.main_image_dims),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            secondary_image_transforms = [
                transforms.RandomCrop(self.secondary_img_dims),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, contrast_factor=0.8))
            ]
            if self.grayscale:
                main_image_transforms.append(transforms.Grayscale(len(image.getbands())))
                secondary_image_transforms.append(transforms.Grayscale(len(image.getbands())))


            main_image = transforms.Compose()(image.copy())

            secondary_images = [
                transforms.Compose()(image.copy()) for i in range(self.max_secondary_images)
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

        if self.is_test:
            return main_image_all.numpy()[:200], secondary_images_all.numpy()[:200], y_all.numpy()[:200]
        else:
            return main_image_all.numpy(), secondary_images_all.numpy(), y_all.numpy()


    def get_plot_name(self, model_name, experiment_config):
        output_file_extension = ".png"
        output_file_name = f"{model_name}_{str(experiment_config)}"
        if self.grayscale:
            output_file_name += f"_{GRAYSCALE}"
        return os.path.join("outputs", self.name, output_file_name + output_file_extension)

    def plot(self, outfile_path): #model_name, active_learning_method):
        #outfile_path = self.get_plot_name(model_name, active_learning_method)
        Path('/'.join(outfile_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)  # create directory if necessary
        # plot the file
        self.tester.plot_results(plot_savename=outfile_path)


    def run_experiments(self):
        # load data
        x_main, x_secondary, y = self.load_dataset()

        for experiment_config in self.experiment_configs:
            for model in self.models:
                # configure the tester for the experiment config
                # do this every time so that the plots will clear
                self.tester = Tester([x_main, x_secondary], y)
                self.tester.INITIAL_TRAIN_DATA_FRACTION = experiment_config.initial_train_data_fraction
                self.tester.ACTIVE_LEARNING_BATCH_SIZE = experiment_config.active_learning_batch_size
                self.tester.TRAINING_EPOCHS = experiment_config.training_epochs
                self.tester.TEST_REPEAT_COUNT = experiment_config.test_repeat_count



                model_name = ""

                for query_function_name in self.query_function_names:
                    try:
                        if query_function_name in self.query_function_name_to_extra_options:
                            for extra_option in self.query_function_name_to_extra_options[query_function_name]:
                                curr_model = model(query_function_name, self.tester.ACTIVE_LEARNING_BATCH_SIZE,
                                                   extra_query_option=extra_option)
                                print("working on model {} with {}".format(curr_model.name(), query_function_name))
                                model_name = curr_model.name()
                                curr_model._name = query_function_name + "_" + extra_option.name()  # this is so that tester will plot it with the correct name

                                self.tester.test_model(curr_model)

                        else:
                            curr_model = model(query_function_name, self.tester.ACTIVE_LEARNING_BATCH_SIZE)
                            print("working on model {} with {}".format(curr_model.name(), query_function_name))
                            model_name = curr_model.name() # this is so that tester will plot it with the correct name
                            curr_model._name = query_function_name


                            self.tester.test_model(curr_model)

                            #curr_model_outfile_name = self.get_plot_name(model_name, experiment_config)
                            #self.plot(curr_model_outfile_name)
                    except Exception as e:
                        print(f"Got exception {e} for model {curr_model.name()} with stack trace:\n{traceback.print_exc()}")

                curr_model_outfile_name = self.get_plot_name(model_name, experiment_config)
                self.plot(curr_model_outfile_name)


    def gradient_embedding_experiment(self):
        """
        This is to try and basically cluster the gradient embeddings and maybe do some PCA to just see what happens
        :return:
        """
        pass
