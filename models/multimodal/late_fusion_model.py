import numpy as np
from test_framework.model_interface import ModelInterface

'''
Defines a wrapper for general late fusion multimodal models
You only need pass a valid list of models (which have train and predict functions fleshed out)
and then pass those as a list to an instance of this model
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
    def __init__(self, models, combination_function, active_learning_function=None,
                 name=None, details=None):
        self.models = models
        self.combination_function = combination_function
        self.active_learning_function = active_learning_function

        self._name = name
        self._details = details

    # IDENTIFIER METHODS
    '''
    Unique name/ID for the model. This may be used by Tester to label plots and performance metrics.
    Returns:
        (str): Name for model
    '''
    def name(self) -> str:
        if self._name is None:
            return "Multimodal Late Fusion with component models " + "_".join(model.name() for model in self.models)
        return self._name

    '''
    Extra details about the model. May be used to specify hyperparameter values, etc which distinguish the
    model from others but don't fit in the shorthand name.
    '''
    def details(self) -> str:
        if self._details is None:
            return "_".join(model.details() for model in self.models)
        return self._details

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
        swapped_axes_train_x = np.swapaxes(train_x, 0,1) # this has shape num_modalities, shape_of_unimodal_data
        for i,model in enumerate(self.models):
            model.train(swapped_axes_train_x[i], train_y)

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
        swapped_axes_test_x = np.swapaxes(test_x, 0, 1)
        return self.combination_function(np.array([self.models[i].predict(swapped_axes_test_x[i]) for i in range(len(self.models))]))

    '''
     Function called to request that the model use its active learning algorithm to choose a subset of
     'unlabeled' samples, which will then be labeled and added to the training set.
     NOTE: The default implementation for this function samples randomly from the unlabeled set.
     Args:
         unlabeled_data (np.ndarray):    Batch of 'unlabeled' inputs from the dataset which are available to be
                                         labeled and added to the training dataset in the next training iteration.
         labeling_batch_size (int):      Number of 'unlabeled' inputs which should be chosen for labeling.
     Returns:
         (np.ndarray):   1-D array of indices into the <unlabeled_data> array, indicating which samples should be
                         labeled and added to training dataset. Shape = (labeling_batch_size,).
     '''

    # NOTE: I would really like to have a better way to handle this, but it's not too obvious what that method is
    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:

        if self.active_learning_function is None:
            raise Exception("If you want to use the default multimodal query function, you need to supply an active learning function upon model instantiation.")

        # default implementation is to use the active learning function provided as model input
        # and call it on the mean of the output probabilities of the models.
        all_preds = []

        swapped_axes_unlabeled_data = np.swapaxes(unlabeled_data, 0, 1)
        for i, model in enumerate(self.models):
            preds = model.predict_proba(
                swapped_axes_unlabeled_data[i].reshape(swapped_axes_unlabeled_data[i].shape[0], -1))
            all_preds.append(preds)

        means_all_preds = np.mean(all_preds, axis=0).reshape(-1, 1)
        return self.active_learning_function(means_all_preds)

