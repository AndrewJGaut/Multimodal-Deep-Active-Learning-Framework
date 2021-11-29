import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import sklearn.metrics as skm

from test_framework.model_interface import ModelInterface
import test_framework.metrics as metrics

'''
Defines the testing framework which stores, trains, and queries a set of active learning models,
and evaluates each model.
'''
class Tester:
    '''
    Args:
        x_data (np.ndarray):    The full set of dataset inputs. First axis is batch, other axes will be untouched
                                and sent to models for training, predicting, querying as is.
        y_data (np.ndarray):    The full set of dataset outputs. First axis is batch, other axes will be untouched
                                and sent to loss function as is.
    '''
    def __init__(self, x_data:np.ndarray, y_data:np.ndarray, training_epochs=10, active_learning_loop_count=10) -> None:
        '''
        --- Default Config ---
        '''

        # Function to call to measure performance (may change for different data formats)
        # Must be consistent for all models compared.
        self.METRIC_FUNCTION = metrics.ACCURACY

        # Fraction of input data to set aside for testing
        self.TEST_DATA_FRACTION = 0.1

        # Number of times to repeat the entire testing process
        # (active learning from start to finish with different initial data)
        self.TEST_REPEAT_COUNT = 1

        # Fraction of training data each model is given to start with (before querying)
        self.INITIAL_TRAIN_DATA_FRACTION = 0.5

        # Number of active learning loops (train, query, repeat) in one test
        self.ACTIVE_LEARNING_LOOP_COUNT = active_learning_loop_count

        # Number of training epochs each model gets in each active learning loop
        self.TRAINING_EPOCHS = training_epochs

        # Number of samples added to training data in each active learning loop
        self.ACTIVE_LEARNING_BATCH_SIZE = 32


        '''
        --- State Variables ---
        '''

        # Store evaluation info (ModelResults) for each ModelInterface object tested
        self.model_results = []



        '''
        --- Data Setup ---
        '''

        # Store original data
        self.x_data = x_data
        self.y_data = y_data

        # Save shuffled data ordering for each test repeat planned
        self.data_orderings = []
        for _ in range(self.TEST_REPEAT_COUNT):
            order = np.random.permutation(self.x_data.shape[0])
            self.data_orderings.append(order)


    
    '''
    Performs active learning tests on the given model, storing both the model and evaluation results.
    Args:
        model (ModelInterface): The model to test (wrapped in the model interface).
    '''
    def test_model(self, model:ModelInterface) -> None:
        model.num_epochs = self.TRAINING_EPOCHS
        results = ModelResults(model)

        for test in range(self.TEST_REPEAT_COUNT):
            # Shuffle data and split into initial train data, 'unlabeled' train data, and test data.
            shuffled_x_data = self.x_data[self.data_orderings[test]]
            shuffled_y_data = self.y_data[self.data_orderings[test]]

            data_size = self.x_data.shape[0]
            test_data_size = round(data_size * self.TEST_DATA_FRACTION)
            initial_train_data_size = round((data_size - test_data_size) * self.INITIAL_TRAIN_DATA_FRACTION)

            test_x = shuffled_x_data[:test_data_size]
            test_y = shuffled_y_data[:test_data_size]
            train_x = shuffled_x_data[test_data_size : test_data_size + initial_train_data_size]
            train_y = shuffled_y_data[test_data_size : test_data_size + initial_train_data_size]
            unlabeled_x = shuffled_x_data[test_data_size + initial_train_data_size :]
            unlabeled_y = shuffled_y_data[test_data_size + initial_train_data_size :]

            # Begin Active Learning Loops
            al_loop_range = trange(self.ACTIVE_LEARNING_LOOP_COUNT)
            al_loop_range.set_description(f"Test {test}")
            for al_loop in al_loop_range:
                # Train Model on current training data
                pre_train_time = time.time()
                al_loop_range.set_description(f"Test {test}")

                # Trigger training epoch
                model.train(train_x, train_y)
                training_time = time.time() - pre_train_time

                # Query model for samples to label
                pre_query_time = time.time()
                label_indices = model.query(unlabeled_x, self.ACTIVE_LEARNING_BATCH_SIZE)
                querying_time = time.time() - pre_query_time

                # Evaluate model on train and test data
                train_performance = self.METRIC_FUNCTION(train_y, model.predict(train_x))
                test_performance = self.METRIC_FUNCTION(test_y, model.predict(test_x))

                # Save model results for this active learning loop
                results.training_performance.append(train_performance)
                results.test_performance.append(test_performance)
                results.data_size.append(train_x.shape[0])
                results.training_time.append(training_time)
                results.querying_time.append(querying_time)

                # Add chosen samples to train data for next AL loop
                train_x = np.concatenate([train_x, unlabeled_x[label_indices]], axis=0)
                train_y = np.concatenate([train_y, unlabeled_y[label_indices]], axis=0)
                unlabeled_x = np.delete(unlabeled_x, label_indices, axis=0)
                unlabeled_y = np.delete(unlabeled_y, label_indices, axis=0)

        # Save results
        self.model_results.append(results)



    '''
    Plots and saves train/test curves for all models, along with their AUC measures
    '''
    def plot_results(self, plot_savename="test_results.png") -> None:
        fig, axarr = plt.subplots(2, 2, figsize=(10,8))

        model_names = [r.model_name for r in self.model_results]

        # Plot training curve
        for r in self.model_results:
            axarr[0][0].plot(r.data_size, r.training_performance, label=r.model_name)
        axarr[0][0].legend()
        axarr[0][0].title.set_text("Training Performance")

        # Plot train AuC measure for each model
        train_auc = []
        for r in self.model_results:
            train_auc.append(skm.auc(range(self.ACTIVE_LEARNING_LOOP_COUNT), r.training_performance))
        axarr[0][1].bar(model_names, train_auc)
        axarr[0][1].title.set_text("Training AuC")

        # Plot testing curve
        for r in self.model_results:
            axarr[1][0].plot(r.data_size, r.test_performance, label=r.model_name)
        axarr[1][0].legend()
        axarr[1][0].title.set_text("Testing Performance")

        # Plot test AuC measure for each model
        test_auc = []
        for r in self.model_results:
            test_auc.append(skm.auc(range(self.ACTIVE_LEARNING_LOOP_COUNT), r.test_performance))
        axarr[1][1].bar(model_names, test_auc)
        axarr[1][1].title.set_text("Testing AuC")

        # Save plots
        plt.savefig(plot_savename)
        plt.show()





'''
Simple struct for Tester to store evaluation info associated with a particular model
'''
class ModelResults:
    def __init__(self, model:ModelInterface):
        # Store name and details to identify model and results if model is lost
        self.model_name = model.name()
        self.model_details = model.details()

        # Store performance on training and test data after each active learning loop
        self.training_performance = []
        self.test_performance = []
        
        # Store data size after each active learning loop
        self.data_size = []

        # Store training time during each active learning loop
        self.training_time = []

        # Store querying time during each active learning loop
        self.querying_time = []
