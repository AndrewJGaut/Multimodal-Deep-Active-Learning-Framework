from __future__ import print_function
from __future__ import division

import numpy as np
from test_framework.model_interface import ModelInterface
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from active_learning.cluster_margin import *
from active_learning.badge import *

# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# for kaggle satellite image classification dataset https://www.kaggle.com/mahmoudreda55/satellite-image-classification
# and then basic active learning was applied.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from utils.pytorch_finetuning_utils import train_model, train_model_given_numpy_arrays, initialize_model
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import sys
sys.path.append("..")
from test_framework.model_interface import ModelInterface
from test_framework.tester import Tester
from utils.data_utils import get_kaggle_satellite_image_classification_dataset_as_numpy_arrays
from active_learning.categorical_query_functions import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model_given_numpy_arrays(model, x1, x2, x2_image_counts, y, criterion, optimizer, num_epochs=25, batch_size=8, verbose=True):

    # Convert data to data loader so that it isn't allocated on the GPU all at once.
    x1_tensor = torch.tensor(x1)
    x2_tensor = torch.tensor(x2)
    x2_image_counts_tensor = torch.tensor(x2_image_counts)
    y_tensor = torch.tensor(y)
    y_tensor = torch.argmax(y_tensor, 1)
    dataset = TensorDataset(x1_tensor,x2_tensor,x2_image_counts_tensor,y_tensor)
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=0,shuffle=True)

    since = time.time()

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for x1,x2,x2_image_counts, labels in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x2_image_counts = x2_image_counts.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(x1,x2,x2_image_counts)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * x1.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(y)
        epoch_acc = running_corrects.double() / len(y)
        if verbose:
            print('Epoch {}/{} - Train Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, num_epochs - 1, epoch_loss, epoch_acc))

    if verbose:
        print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model



'''
Defines a wrapper for a middle fusion model based on AlexNet
'''


# https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://pytorch.org/vision/master/_modules/torchvision/models/alexnet.html

class MiddleFusionNet(torch.nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.5) -> None:
        super(MiddleFusionNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
        )
        self.final_linear = nn.Linear(1024*2, num_classes)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x2_image_counts: torch.Tensor) -> torch.Tensor:
        x1_features = self.features(x1)
        x1_features = self.avgpool(x1_features)
        x1_features = torch.flatten(x1_features, 1)
        x1_embeddings = self.classifier(x1_features)

        x2_embeddings = []
        for count, x2_image_stack in zip(x2_image_counts,x2):
            if count > 0:
                cur_x2_features = self.features(x2_image_stack[0:count])
                cur_x2_features = self.avgpool(cur_x2_features)
                cur_x2_features = torch.flatten(cur_x2_features, 1)
                cur_x2_embeddings = self.classifier(cur_x2_features)
                x2_embeddings.append(torch.mean(cur_x2_embeddings,axis=0))
            else:
                x2_embeddings.append(torch.zeros(1024).to(device))
        x2_embeddings = torch.stack(x2_embeddings)

        all_embeddings = torch.cat((x1_embeddings,x2_embeddings),1)
        output = self.final_linear(all_embeddings)
        return output
    def details(self):
        return "a middle fusion model based on AlexNet"

class MiddleFusionModel(ModelInterface):
    '''
    Instantiate the middle fusion model.
    '''
    def __init__(self, query_function_name,
                 name=None, details=None,
                num_epochs=3, batch_size=8, train_verbose=True,
                query_function=None):
        self.model = MiddleFusionNet(num_classes=4)
        self._name = name
        self._details = details

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self._name = name
        self._details = details
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_verbose = train_verbose
        self.query_function = query_function

        params_to_update = self.model.parameters()

        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        self._optimizer = optimizer_ft

        # store criterion
        self._criterion = nn.CrossEntropyLoss()



        self.query_function_name = query_function_name
        # For cluster-margin active learning algorithm, clusters must be
        # saved between queries
        if self.query_function_name == "CLUSTER_MARGIN":
            raise NotImplementedError("Cluster margin is not yet implemented for late fusion")
            """
            self.cluster_margin = ClusterMarginQueryFunction(
                self.model, None, # [self.model.post_fusion_layer.weight],
                margin_batch_size=2 * self.batch_size,
                target_batch_size=self.batch_size
            )
            self.badge = None
            """
        elif self.query_function_name == "BADGE":
            self.badge = BADGEQueryFunction(
                self.model, None,
                margin_batch_size=2 * self.batch_size,
                target_batch_size=self.batch_size
            )
            self.cluster_margin = None
        else:
            self.cluster_margin = None
            self.badge = None






    # IDENTIFIER METHODS
    '''
    Unique name/ID for the model. This may be used by Tester to label plots and performance metrics.
    Returns:
        (str): Name for model
    '''
    def name(self) -> str:
        if self._name is None:
            return "Multimodal middle fusion model based on AlexNet"
        return self._name

    '''
    Extra details about the model. May be used to specify hyperparameter values, etc which distinguish the
    model from others but don't fit in the shorthand name.
    '''
    def details(self) -> str:
        return self.model.details()

    def train(self, x1: np.ndarray, x2: np.ndarray, x2_image_counts: np.ndarray, y: np.ndarray) -> None:
        # x1,x2 = np.swapaxes(train_x,0,1)
        self.model = train_model_given_numpy_arrays(self.model, x1, x2, x2_image_counts, y,
                                                    self._criterion, self._optimizer,
                                                    self.num_epochs, self.batch_size, verbose=self.train_verbose)

    def predict(self, x1: np.ndarray, x2: np.ndarray, x2_image_counts: np.ndarray):
        self.model.eval()
        # x1,x2 = np.swapaxes(test_x,0,1)
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        x2_image_counts = torch.tensor(x2_image_counts)
        dataset = TensorDataset(x1,x2,x2_image_counts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
        preds_list = []
        for (x1,x2,x2_image_counts,) in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x2_image_counts = x2_image_counts.to(device)
            preds_list.append(self.model(x1,x2,x2_image_counts).cpu().detach().numpy())
        return np.vstack(preds_list)

    def predict_proba(self, x1: np.ndarray, x2: np.ndarray, x2_image_counts: np.ndarray) -> np.ndarray:
        self.model.eval()
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        softmax_outputs = softmax(self.predict(x1, x2, x2_image_counts))
        return softmax_outputs

    """
    def query(self, x1: np.ndarray, x2: np.ndarray, x2_image_counts: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        softmax_outputs = self.predict_proba(x1, x2, x2_image_counts)
        indices = self.active_learning_function(softmax_outputs, labeling_batch_size)
        return indices
    """
    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        #softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        #softmax_outputs = softmax(self.predict(unlabeled_data))

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


class ActiveLearningModel(ModelInterface):
    def __init__(self, query_function_name, feature_extract=True,
                 num_epochs=8, batch_size=32, train_verbose=True):

        #model_ft, input_size = initialize_model("squeezenet", 4, feature_extract, use_pretrained=True)
        self.model = MiddleFusionModel #model_ft
        #self.model.to(device)

        self.query_function_name = query_function_name

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = self.model.parameters()
        verbose = False
        if verbose:
            print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    if verbose:
                        print("\t", name)
        elif verbose:
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        self._optimizer = optimizer_ft

        # store criterion
        self._criterion = nn.CrossEntropyLoss()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_verbose = train_verbose

        # For cluster-margin active learning algorithm, clusters must be
        # saved between queries
        if self.query_function_name == "CLUSTER_MARGIN":
            raise NotImplementedError("Cluster margin not yet implemented for middle fusion model")
            """
            self.cluster_margin = ClusterMarginQueryFunction(
                self.model, [self.model.post_fusion_layer.weight],
                margin_batch_size=2 * self.active_learning_batch_size,
                target_batch_size=self.active_learning_batch_size
            )
            self.badge = None
            """
        elif self.query_function_name == "BADGE":
            self.badge = BADGEQueryFunction(
                self.model, None,
                margin_batch_size=2 * self.batch_size,
                target_batch_size=self.batch_size
            )
            self.cluster_margin = None
        else:
            self.cluster_margin = None
            self.badge = None

    def name(self) -> str:
        return self.model.name

    def details(self) -> str:
        return self.model.details

    def train(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.model = train_model_given_numpy_arrays(self.model, train_x, train_y, self._criterion, self._optimizer,
                                                    self.num_epochs, self.batch_size, verbose=self.train_verbose)

    def predict(self, test_x: np.ndarray):
        self.model.eval()
        x_tensor = torch.tensor(test_x)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
        preds_list = []
        for (inputs,) in dataloader:
            inputs = inputs.to(device)
            preds_list.append(self.model(inputs).cpu().detach().numpy())
        return np.vstack(preds_list)

    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        #softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        #softmax_outputs = softmax(self.predict(unlabeled_data))

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

        #indices = QUERY_FUNCTION(softmax_outputs, labeling_batch_size)
        #return indices