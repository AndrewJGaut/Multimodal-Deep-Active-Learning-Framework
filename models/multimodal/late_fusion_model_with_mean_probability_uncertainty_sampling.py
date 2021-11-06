from models.multimodal.late_fusion_model import LateFusionModel
import numpy as np


class LateFusionModelWithMeanProbabilityUncertaintySampling(LateFusionModel):

    # we'll just take the mean over the confidence values
    def query(self, unlabeled_data: np.ndarray, labeling_batch_size: int) -> np.ndarray:
        all_preds = []

        swapped_axes_unlabeled_data = np.swapaxes(unlabeled_data, 0, 1)
        for i,model in enumerate(self.models):
            preds = model.clf.predict(swapped_axes_unlabeled_data[i].reshape(swapped_axes_unlabeled_data[i].shape[0], -1))
            all_preds.append(preds)

        all_preds = np.array(all_preds)
        means_all_preds = np.mean(all_preds, axis=0).reshape(-1,1)
        least_confident_pred_per_x = 1 - np.max(means_all_preds, axis=1)
        least_confident_pred_indices = np.argpartition(least_confident_pred_per_x, -1 * labeling_batch_size)[
                                       -1 * labeling_batch_size:].reshape(-1,1)

        return least_confident_pred_indices
