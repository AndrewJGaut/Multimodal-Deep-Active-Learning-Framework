from typing import List
import numpy as np
from tqdm.notebook import trange as trange
import matplotlib.pyplot as plt
%matplotlib inline

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
from active_learning.badge import *



# Model Constants
FINAL_LAYER_LEN = 64

# Training Constants
TEST_DATA_FRACTION = 0.05

class MultiModalLateFusionModel(nn.Module):
    def __init__(self):
        super(MultiModalLateFusionModel, self).__init__()

        # Data Constants
        self.NUM_CLASSES = 4
        self.MAIN_IMG_DIMS = (64, 64)
        self.SECONDARY_IMG_DIMS = (32, 32)
        self.MAX_SECONDARY_IMAGES = 5

        # Simple convolutional network operates on original full images,
        # and outputs a set of features useful for a final classification layer
        self.main_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * (self.MAIN_IMG_DIMS[0] // 2 ** 3) * (self.MAIN_IMG_DIMS[1] // 2 ** 3), FINAL_LAYER_LEN // 2),
            nn.Tanh()
        )

        # This secondary branch operates on cropped parts of the image with
        # increased contrast. If there are multiple secondary images, we use
        # the average of the features output by this branch
        self.secondary_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * (self.SECONDARY_IMG_DIMS[0] // 2 ** 3) * (self.SECONDARY_IMG_DIMS[1] // 2 ** 3), FINAL_LAYER_LEN // 2),
            nn.Tanh()
        )



        # Single dense layer which combines information directly after fusing both model modes
        self.post_fusion_layer = nn.Linear(FINAL_LAYER_LEN, self.NUM_CLASSES)

    '''
    Args:
        main_image (torch.Tensor):          Shape = (batch_size, 3, *(MAIN_IMG_DIMS))
        secondary_images (torch.Tensor):    Variable size list of cropped high-contrast parts
                                            of the main image. The list is padded to the max
                                            length with images of all zeros.
                                            Shape = (batch_size, MAX_SECONDARY_IMAGES, 3, *(SECONDARY_IMG_DIMS))
    '''

    def forward(self, main_image: torch.Tensor, secondary_images: torch.Tensor) -> torch.Tensor:
        batch_size = main_image.shape[0]

        # Compute main branch
        main_features = self.main_branch(main_image)

        # Count the number of secondary images
        slot_has_image = (secondary_images != torch.zeros(3, *self.SECONDARY_IMG_DIMS).to(DEVICE)).reshape(batch_size,
                                                                                                      self.MAX_SECONDARY_IMAGES,
                                                                                                      -1).prod(dim=-1,
                                                                                                               keepdim=True)  # shape = (batch_size, MAX_SECONDARY_IMAGES, 1)
        image_count_per_sample = slot_has_image.sum(dim=1)  # shape = (batch_size, 1)

        # Compute features for every possible secondary image
        all_secondary_images = secondary_images.reshape(batch_size * self.MAX_SECONDARY_IMAGES, 3, *self.SECONDARY_IMG_DIMS)
        all_secondary_features = self.secondary_branch(all_secondary_images).reshape(batch_size, self.MAX_SECONDARY_IMAGES,
                                                                                     -1)

        # Mask out features for secondary image slots that didn't contain an actual image (i.e. were all zeros)
        masked_secondary_image_features = all_secondary_features * slot_has_image

        # Average the secondary branch features of all included secondary branches
        averaged_secondary_image_features = masked_secondary_image_features.sum(dim=1) / image_count_per_sample

        # Concatenate both feature modes
        fused_features = torch.cat([main_features, averaged_secondary_image_features], dim=1)

        # Pass through final layer
        output_logits = self.post_fusion_layer(fused_features)

        output_probabilities = torch.softmax(output_logits, dim=1)
        return output_probabilities


# This interface uses the multimodal model above with a specified query function
class MultiModalLateFusionModelInterface(ModelInterface):
    def __init__(self, query_function_name: str, active_learning_batch_size: int = 32):
        self.query_function_name = query_function_name
        self.active_learning_batch_size = active_learning_batch_size

        # Model Training Constants
        self.TRAINING_MINIBATCH_SIZE = 128

        self.reset()

    def name(self):
        return "Multimodal late fusion model based on AlexNet" #+ self.query_function_name

    def reset(self):
        self.model = MultiModalLateFusionModel().to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # For cluster-margin active learning algorithm, clusters must be
        # saved between queries
        if self.query_function_name == "CLUSTER_MARGIN":
            self.cluster_margin = ClusterMarginQueryFunction(
                self.model, [self.model.post_fusion_layer.weight],
                margin_batch_size=2 * self.active_learning_batch_size,
                target_batch_size=self.active_learning_batch_size
            )
            self.badge = None
        elif self.query_function_name == "BADGE":
            self.badge = BADGEQueryFunction(
                self.model, [self.model.post_fusion_layer.weight],
                margin_batch_size=2 * self.active_learning_batch_size,
                target_batch_size=self.active_learning_batch_size
            )
            self.cluster_margin = None
        else:
            self.cluster_margin = None
            self.badge = None

    def train(self, train_x: List[np.ndarray], train_y: np.ndarray) -> None:
        data_len = train_y.shape[0]

        # Extract specific data modes
        x_main, x_secondary = train_x

        # Pass through all minibatches in training set, but skip any partial minibatch at the end
        batch_start = 0
        while batch_start + self.TRAINING_MINIBATCH_SIZE < data_len:
            x_main_minibatch = torch.from_numpy(x_main[batch_start: batch_start + self.TRAINING_MINIBATCH_SIZE]).to(
                DEVICE)
            x_secondary_minibatch = torch.from_numpy(
                x_secondary[batch_start: batch_start + self.TRAINING_MINIBATCH_SIZE]).to(DEVICE)
            y_minibatch = torch.from_numpy(train_y[batch_start: batch_start + self.TRAINING_MINIBATCH_SIZE]).to(DEVICE)

            # Train on minibatch
            self.opt.zero_grad()

            model_output = self.model(
                x_main_minibatch,
                x_secondary_minibatch
            )

            # Compute cross-entropy loss
            loss = -torch.mean(torch.sum(y_minibatch * torch.log(model_output + 1e-9), dim=1))
            loss.backward()
            self.opt.step()

            # Iterate minibatch
            batch_start += self.TRAINING_MINIBATCH_SIZE

    def predict(self, test_x: List[np.ndarray]) -> np.ndarray:
        data_len = test_x[0].shape[0]
        output = np.zeros((data_len, self.NUM_CLASSES))

        with torch.no_grad():
            # Extract specific data modes
            x_main, x_secondary = test_x

            # Iterate through minibatches without skipping partial ending
            batch_start = 0
            while batch_start < data_len:
                current_minibatch_size = min(self.TRAINING_MINIBATCH_SIZE, data_len - batch_start)

                # Convert to tensors
                x_main_minibatch = torch.from_numpy(x_main[batch_start: batch_start + current_minibatch_size]).to(
                    DEVICE)
                x_secondary_minibatch = torch.from_numpy(
                    x_secondary[batch_start: batch_start + current_minibatch_size]).to(DEVICE)

                predictions = self.model(
                    x_main_minibatch,
                    x_secondary_minibatch
                )
                output[batch_start: batch_start + current_minibatch_size] = predictions.cpu().numpy()

                # Iterate minibatch
                batch_start += current_minibatch_size

        return output

    def query(self, unlabeled_data: List[np.ndarray], labeling_batch_size: int) -> np.ndarray:
        if self.query_function_name == "RANDOM":
            data_size = unlabeled_data[0].shape[0]
            return np.random.choice(np.arange(data_size), size=labeling_batch_size, replace=False)

        if self.query_function_name == "MIN_MAX":
            return MIN_MAX(self.predict(unlabeled_data), labeling_batch_size)

        if self.query_function_name == "MIN_MARGIN":
            return MIN_MARGIN(self.predict(unlabeled_data), labeling_batch_size)

        if self.query_function_name == "MAX_ENTROPY":
            return MAX_ENTROPY(self.predict(unlabeled_data), labeling_batch_size)

        if self.query_function_name == "CLUSTER_MARGIN":
            return self.cluster_margin.query(unlabeled_data)

        if self.query_function_name == "BADGE":
            return self.badge.query(unlabeled_data)

        raise ValueError(f"Unrecognized query function name: {self.query_function_name}")

