from experiments.experiment import Experiment, ExperimentConfig, get_experiment_configs
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
    initial_train_data_fractions = [0.001, 0.001, 0.005, 0.01, 0.05]
    active_learning_batch_sizes = [32, 64, 32, 32, 32, 32]
    training_epochs = [20, 20, 18, 15, 15]
    test_repeat_counts = [10, 10, 8, 8, 5]
    experiment_configs = get_experiment_configs(initial_train_data_fractions, active_learning_batch_sizes,
                                                training_epochs, test_repeat_counts)
    exp = Experiment(models=models, query_function_names=query_function_names,
                     query_function_name_to_extra_options=options, experiment_configs=experiment_configs)
    exp.run_experiments()