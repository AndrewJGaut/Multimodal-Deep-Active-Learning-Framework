import numpy as np
from test_framework.model_interface import ModelInterface

'''
Defines the wrapper interface through which a Tester object can store particular models,
trigger model training, and query for active learning.
'''


class LateFusionModel(ModelInterface):
    '''
    Instantiate a late fusion model with its unimodal components being the models in this list.
    Parameters:
            (list): list of objects that implement the ModelInterface.
                    These will be used as part of the prediction method
            ((*args) -> np.ndarray: this function will be used to combine predictions from the models
                                    E.g., if we used the MEAN function, then preditions would be averaged. (For a probability distribution, this would have to be renormalized)
    '''
    def __init__(self, models, combination_function):
        self.models = models
        self.combination_function = combination_function

    # IDENTIFIER METHODS
    '''
    Unique name/ID for the model. This may be used by Tester to label plots and performance metrics.
    Returns:
        (str): Name for model
    '''
    def name(self) -> str:
        return "_".join(model.name() for model in self.models)

    '''
    Extra details about the model. May be used to specify hyperparameter values, etc which distinguish the
    model from others but don't fit in the shorthand name.
    '''
    def details(self) -> str:
        return "_".join(model.details() for model in self.models)

    # INTERACTION METHODS
    '''
    Function called to request that the model train on the given training dataset. Training should pass
    through the dataset only once (1 epoch). Testing framework will call repeatedly to achieve multiple epochs.
    Args:
        train_x (np.ndarray):       Training inputs, with batch as the first axis.
                                    In this case, these should have shape len(self.models), shape_of_data_for_each_modality
                                    In other words, the index into axis0 gives the index of the modality
        train_y (np.ndarray):       Training outputs for supervised learning, with batch as the first axis.
                                    Should have shape (num_examples)
    Returns:
        (float):    Average training loss
    '''
    def train(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        for i,model in enumerate(self.models):
            model.train(train_x[i], train_y)

    '''
    Function called to request that the model predict outputs for the given val/test dataset.
    Args:
        test_x (np.ndarray):    Testing inputs, with batch as the first axis.
                                Should have shape len(self.models),shape_of_each_modality
    Returns:
        (np.ndarray):           Network outputs, with batch as the first axis, corresponding to each sample
                                in test_x.
    '''

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        # note: I think there's a better way to create the np array passed an argument below.
        return self.combination_function(np.array([self.models[i].predict(test_x[i]) for i in range(len(self.models))]))
