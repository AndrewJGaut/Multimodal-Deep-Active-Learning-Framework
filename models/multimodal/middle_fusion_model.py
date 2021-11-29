import numpy as np
from test_framework.model_interface import ModelInterface
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model_given_numpy_arrays(model, x1, x2, y, criterion, optimizer, num_epochs=25, batch_size=8, is_inception=False, verbose=True):
    if is_inception:
        raise NotImplementedError("training using tensors not supported for inception model")

    # Convert data to data loader so that it isn't allocated on the GPU all at once.
    x1_tensor = torch.tensor(x1)
    x2_tensor = torch.tensor(x2)
    y_tensor = torch.tensor(y)
    y_tensor = torch.argmax(y_tensor, 1)
    dataset = TensorDataset(x1_tensor,x2_tensor,y_tensor)
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=0,shuffle=True)

    since = time.time()

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for x1,x2, labels in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(x1,x2)
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

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6 * 2, 4096), # added *2
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x_satellite: torch.Tensor, x_streetlevel: torch.Tensor) -> torch.Tensor:
        features_satellite = self.features(x_satellite)
        features_satellite = self.avgpool(features_satellite)
        features_satellite = torch.flatten(features_satellite, 1)

        features_streetlevel = self.features(x_streetlevel)
        features_streetlevel = self.avgpool(features_streetlevel)
        features_streetlevel = torch.flatten(features_streetlevel, 1)
        
        all_features = torch.cat((features_satellite,features_streetlevel),1)

        output = self.classifier(all_features)
        return output
    def details(self):
        return "a middle fusion model based on AlexNet"

class MiddleFusionModel(ModelInterface):
    '''
    Instantiate the middle fusion model.
    '''
    def __init__(self, active_learning_function,
                 name=None, details=None,
                num_epochs=3, batch_size=8, train_verbose=True,
                query_function=None):
        self.active_learning_function = active_learning_function
        self.model = MiddleFusionNet(num_classes=4)
        self._name = name
        self._details = details

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def train(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        x1,x2 = np.swapaxes(train_x,0,1)
        self.model = train_model_given_numpy_arrays(self.model, x1, x2, train_y, self._criterion, self._optimizer,
                                                    self.num_epochs, self.batch_size, verbose=self.train_verbose)

    def predict(self, test_x: np.ndarray):
        self.model.eval()
        x1,x2 = np.swapaxes(test_x,0,1)
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        dataset = TensorDataset(x1,x2)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
        preds_list = []
        for (x1,x2,) in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            preds_list.append(self.model(x1,x2).cpu().detach().numpy())
        return np.vstack(preds_list)

    def predict_proba(self, test_x: np.ndarray) -> np.ndarray:
        self.model.eval()
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        softmax_outputs = softmax(self.predict(test_x))
        return softmax_outputs

    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        softmax_outputs = self.predict_proba(unlabeled_data)
        indices = self.active_learning_function(softmax_outputs, labeling_batch_size)
        return indices
