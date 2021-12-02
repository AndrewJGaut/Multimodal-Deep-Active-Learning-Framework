import numpy as np
from torch import random
from test_framework.model_interface import ModelInterface
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from typing import List

from active_learning.categorical_query_functions import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model_given_numpy_arrays(model, x1, x2, x2_image_counts, y, criterion, optimizer, batch_size=8, verbose=True):
    NUM_EPOCHS = 1 # the number of training epochs is set in the Tester class

    # Convert data to data loader so that it isn't allocated on the GPU all at once.
    x1_tensor = torch.tensor(x1)
    x2_tensor = torch.tensor(x2)
    x2_image_counts_tensor = torch.tensor(x2_image_counts)
    y_tensor = torch.tensor(y)
    y_tensor = torch.argmax(y_tensor, 1)
    dataset = TensorDataset(x1_tensor,x2_tensor,x2_image_counts_tensor,y_tensor)
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=0,shuffle=True)

    since = time.time()

    for epoch in range(NUM_EPOCHS):
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
                epoch, NUM_EPOCHS - 1, epoch_loss, epoch_acc))

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
    def __init__(self, num_classes: int = 4, dropout: float = 0.5, random_seed=None) -> None:
        if random_seed is not None:
            torch.manual_seed(random_seed)
        super(MiddleFusionNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d((6, 6))
        self.linear1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((6, 6))
        self.linear2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
        )
        self.final_linear = nn.Linear(1024*2, num_classes)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x2_image_counts: torch.Tensor) -> torch.Tensor:
        x1_features = self.conv1(x1)
        x1_features = self.avgpool1(x1_features)
        x1_features = torch.flatten(x1_features, 1)
        x1_embeddings = self.linear1(x1_features)

        x2_embeddings = []
        for count, x2_image_stack in zip(x2_image_counts,x2):
            if count > 0:
                cur_x2_features = self.conv2(x2_image_stack[0:count])
                cur_x2_features = self.avgpool2(cur_x2_features)
                cur_x2_features = torch.flatten(cur_x2_features, 1)
                cur_x2_embeddings = self.linear2(cur_x2_features)
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
    def __init__(self, query_function_name, active_learning_batch_size = 32,
                 random_seed=None, train_verbose=True):
        self.query_function_name = query_function_name
        self.model = MiddleFusionNet(num_classes=4,random_seed=random_seed)
        self._name = self.model.name()
        self._details = self.model.details()

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.train_verbose = train_verbose

        params_to_update = self.model.parameters()

        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        self._optimizer = optimizer_ft

        # store criterion
        self._criterion = nn.CrossEntropyLoss()


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

    def train(self, x:List[np.ndarray], y: np.ndarray) -> None:
        x1, x2, x2_image_counts = x
        batch_size = x.shape[0] # infer batch size from input
        torch.manual_seed(0) # comment this line out to increase variation across experiments
        self.model = train_model_given_numpy_arrays(self.model, x1, x2, x2_image_counts, y,
                                                    self._criterion, self._optimizer,
                                                    1, batch_size, verbose=self.train_verbose)


    def predict(self, x:List[np.ndarray]):
        x1, x2, x2_image_counts = x
        self.model.eval()
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

    def predict_proba(self, x:List[np.ndarray]) -> np.ndarray:
        self.model.eval()
        softmax = lambda s: np.exp(s) / np.sum(np.exp(s), axis=-1, keepdims=True)
        softmax_outputs = softmax(self.predict(x))
        return softmax_outputs

    def query(self, unlabeled_x:List[np.ndarray], labeling_batch_size: int) -> np.ndarray:
        if self.query_function_name == "RANDOM":
            return RANDOM(self.predict_proba(unlabeled_x), labeling_batch_size)
        elif self.query_function_name == "MIN_MAX":
            return MIN_MAX(self.predict_proba(unlabeled_x), labeling_batch_size)
        elif self.query_function_name == "MIN_MARGIN":
            return MIN_MARGIN(self.predict_proba(unlabeled_x), labeling_batch_size)
        elif self.query_function_name == "MAX_ENTROPY":
            return MAX_ENTROPY(self.predict_proba(unlabeled_x), labeling_batch_size)
        elif self.query_function_name == "CLUSTER_MARGIN":
            # return self.cluster_margin.query(unlabeled_data)
            pass
        elif self.query_function_name == "BADGE":
            #return self.badge.query(unlabeled_data)
            pass
        else:
            raise ValueError(f"Unrecognized query function name: {self.query_function_name}")