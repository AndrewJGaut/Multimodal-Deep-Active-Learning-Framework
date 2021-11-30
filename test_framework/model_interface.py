import numpy as np
from typing import List

'''
Defines the wrapper interface through which a Tester object can store particular models, 
trigger model training, and query for active learning.
'''
class ModelInterface:
    # IDENTIFIER METHODS
    '''
    Unique name/ID for the model. This may be used by Tester to label plots and performance metrics.
    Returns:
        (str): Name for model
    '''
    def name(self) -> str:
        return "ModelInterface (Unimplemented)"
    
    '''
    Extra details about the model. May be used to specify hyperparameter values, etc which distinguish the
    model from others but don't fit in the shorthand name.
    '''
    def details(self) -> str:
        return "(Unimplemented)"
    

    # INTERACTION METHODS
    '''
    Function called to request that the model train on the given training dataset. Training should pass
    through the dataset only once (1 epoch). Testing framework will call repeatedly to achieve multiple epochs.
    Args:
        train_x ([np.ndarray]):     List of modes of training inputs, with batch as the first axis for each.
        train_y (np.ndarray):       Training outputs for supervised learning, with batch as the first axis.
    Returns:
        (float):    Average training loss 
    '''
    def train(self, train_x:List[np.ndarray], train_y:np.ndarray) -> None:
        raise NotImplementedError

    '''
    Function called to request that the model predict outputs for the given val/test dataset.
    Args:
        test_x ([np.ndarray]):  List of modes for test inputs, with batch as the first axis for each.
    Returns:
        (np.ndarray):           Network outputs, with batch as the first axis, corresponding to each sample 
                                in test_x.
    '''
    def predict(self, test_x:List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    '''
    Function called to request that the model use its active learning algorithm to choose a subset of
    'unlabeled' samples, which will then be labeled and added to the training set.
    NOTE: The default implementation for this function samples randomly from the unlabeled set.
    Args:
        unlabeled_data ([np.ndarray]):  List of modes for batch of 'unlabeled' inputs from the dataset which are available to be 
                                        labeled and added to the training dataset in the next training iteration.
        labeling_batch_size (int):      Number of 'unlabeled' inputs which should be chosen for labeling.
    Returns:
        (np.ndarray):   1-D array of indices into the <unlabeled_data> array, indicating which samples should be
                        labeled and added to training dataset. Shape = (labeling_batch_size,).
    '''
    def query(self, unlabeled_data:List[np.ndarray], labeling_batch_size:int) -> np.ndarray:
        data_size = unlabeled_data[0].shape[0]
        return np.random.choice(np.arange(data_size), size=labeling_batch_size, replace=False)

    '''
    Resets the model, for running multiple tests in sequence.
    '''
    def reset(self) -> None:
        raise NotImplementedError