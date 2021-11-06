import numpy as np
from test_framework.model_interface import ModelInterface
import sklearn



class ProbabilisticSVM(ModelInterface):
    #clf = svm.SVC(probability=True)
    clf = sklearn.linear_model.SGDClassifier(loss="log")

    def name(self) -> str:
        return "SVM"

    def details(self) -> str:
        return "default_hyperparameters"

    def train(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        if len(train_x.shape) > 2:
            train_x = train_x.reshape(train_x.shape[0], -1)
        self.clf.fit(train_x, train_y)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        if len(test_x.shape) > 2:
            test_x = test_x.reshape(test_x.shape[0], -1)
        pred_probabilities = self.clf.predict_proba(test_x)
        return np.argmax(pred_probabilities, axis=1)

    """
    This implements the basic uncertainty sampling method.
    This SVM outputs probabilistic predictions (i.e, p(y|x) over all y labels)
    So, we find: x* = argmax_x {1 - argmax_y{p(y|x)}} (the x's with the least confident prediction for the highest class)
    """
    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        preds = self.clf.predict_proba(unlabeled_data)
        least_confident_pred_per_x = 1 - np.max(preds, axis=1)
        least_confident_pred_indices = np.argpartition(least_confident_pred_per_x, -1 * labeling_batch_size)[
                                       -1 * labeling_batch_size:]

        return least_confident_pred_indices

