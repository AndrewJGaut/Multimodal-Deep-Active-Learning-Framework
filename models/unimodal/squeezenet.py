import numpy as np
from test_framework.model_interface import ModelInterface
import torch
import torch.optim as optim
import torch.nn as nn
from utils.pytorch_finetuning_utils import *


class SqueezeNet(ModelInterface):
    def __init__(self, num_dataset_classes=4, name="no name provided", details="no details provided", feature_extract=True,
                 num_epochs=3, batch_size=8, train_verbose=True,
                 query_function=None):
        self.model, _ = initialize_model("squeezenet", num_dataset_classes, feature_extract, use_pretrained=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self._name = name
        self._details = details
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_verbose = train_verbose
        self.query_function = query_function

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

    def name(self) -> str:
        return self._name

    def details(self) -> str:
        return self._details

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

    def predict_proba(self, test_x: np.ndarray) -> np.ndarray:
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        softmax_outputs = softmax(self.predict(test_x))

    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        softmax_outputs = self.predict_proba(unlabeled_data)
        indices = self.query_function(softmax_outputs, labeling_batch_size)
        return indices