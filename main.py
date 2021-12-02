from experiments.experiment import Experiment
from models.multimodal.late_fusion_model_torch import MultiModalLateFusionModelInterface
from models.multimodal.middle_fusion_model import ActiveLearningModel
from models.multimodal.middle_fusion_model import MiddleFusionModel
from clustering.cluster import KMeans, KMeansPlusPlus, AgglomerativeCluster, GMM
from clustering.sampling import KMeansPlusPlusSeeding, WeightedKMeansSampling

if __name__ == '__main__':
    """
    We'll run all the experiments in this function
    """
    models = [MultiModalLateFusionModelInterface, ActiveLearningModel]
    #models = [MiddleFusionModel]
    query_function_names = ["RANDOM", "MIN_MAX", "MIN_MARGIN", "MAX_ENTROPY", "CLUSTER_MARGIN", "BADGE"]
    options = {
        "CLUSTER_MARGIN": [KMeans, KMeansPlusPlus, AgglomerativeCluster, GMM],
        "BADGE": [KMeansPlusPlusSeeding, WeightedKMeansSampling]
    }
    exp = Experiment(models=models, query_function_names=query_function_names,
                     query_function_name_to_extra_options=options, active_learning_batch_size=32,
                     training_epochs=10, test_repeat_count=4)
    exp.run_experiments()