from experiments.experiment import Experiment, ExperimentConfig, get_experiment_configs
from models.multimodal.late_fusion_model_torch import MultiModalLateFusionModelInterface
from models.multimodal.middle_fusion_model import MiddleFusionModel
from clustering.cluster import SklearnKMeans, SklearnAgglomerativeCluster, SklearnGMM
from clustering.sampling import KMeansPlusPlusSeeding, WeightedKMeansSampling

from models.multimodal.late_fusion_model_linear_torch import MultiModalLateFusionLinearModelInterface

ALL_MODELS = [MiddleFusionModel, MultiModalLateFusionModelInterface, MultiModalLateFusionLinearModelInterface]
ALL_QUERY_FUNCTION_NAMES = ["RANDOM", "MIN_MAX", "MIN_MARGIN", "MAX_ENTROPY", "CLUSTER_MARGIN", "BADGE"]
ALL_OPTIONS = {
        "CLUSTER_MARGIN": [SklearnKMeans(), SklearnAgglomerativeCluster(), SklearnGMM()],
        "BADGE": [KMeansPlusPlusSeeding(), WeightedKMeansSampling()]
    }

BASELINE_CONFIGS = [
    ExperimentConfig(
        initial_train_data_fraction=0.0005, # start with two data points (this fraction works for this b/c size of our dataset is fixed)
        active_learning_batch_size=1,
        training_epochs=20,
        test_repeat_count=8 #8
    ),
    ExperimentConfig(
        initial_train_data_fraction=0.05,
        active_learning_batch_size=32,
        training_epochs=20,
        test_repeat_count= 8 # 8
    )
]



other_experiment_config = ExperimentConfig(
        initial_train_data_fraction=0.05,
        active_learning_batch_size=64,
        training_epochs=20,
        test_repeat_count=2
)

big_batch_size_config = ExperimentConfig(
        initial_train_data_fraction=0.05,
        active_learning_batch_size=256,
        training_epochs=20,
        test_repeat_count=2
)

def run_all_relevant_experiments():
    new_experiments()
    cluster_margin_cluster_methods_experiments()
    badge_sample_methods_experiments()
    new_experiments(grayscale=True)



def first_experiment(grayscale=False):
    initial_train_data_fractions = [0.001, 0.001, 0.005, 0.01, 0.05]
    active_learning_batch_sizes = [32, 64, 32, 32, 32, 32]
    training_epochs = [20, 20, 18, 15, 15]
    test_repeat_counts = [10, 10, 8, 8, 5]
    experiment_configs = get_experiment_configs(initial_train_data_fractions, active_learning_batch_sizes,
                                                training_epochs, test_repeat_counts)

    exp = Experiment(name="first_experiment", models=ALL_MODELS, query_function_names=ALL_QUERY_FUNCTION_NAMES,
                     query_function_name_to_extra_options=ALL_OPTIONS, experiment_configs=experiment_configs,
                     grayscale=grayscale)
    exp.run_experiments()

def recreate_late_fusion_notebook_experiment():
    exp = Experiment(name="recreate_late_fusion_notebook_experiment",
                     models=[MultiModalLateFusionModelInterface],
                     query_function_names=ALL_QUERY_FUNCTION_NAMES,
                     experiment_configs=[ExperimentConfig(
                         initial_train_data_fraction=0.05, final_model_layer_len=64,
                         active_learning_batch_size=256, training_epochs=4, test_repeat_count=2
                     )])
    exp.run_experiments()


def very_quick_test():
    exp = Experiment(name="very_quick_test",
                     models=[MultiModalLateFusionLinearModelInterface],
                     query_function_names=ALL_QUERY_FUNCTION_NAMES,
                     experiment_configs=[ExperimentConfig(
                         initial_train_data_fraction=0.001, final_model_layer_len=64,
                         active_learning_batch_size=32, training_epochs=2, test_repeat_count=2
                     ),
                         ExperimentConfig(
                             initial_train_data_fraction=0.001, final_model_layer_len=64,
                             active_learning_batch_size=16, training_epochs=1, test_repeat_count=1
                         )
                     ],
                     is_test=True)
    exp.run_experiments()

def very_quick_test_grayscale():
    exp = Experiment(name="very_quick_test_grayscale",
                     models=[MiddleFusionModel],
                     query_function_names=ALL_QUERY_FUNCTION_NAMES,
                     experiment_configs=[ExperimentConfig(
                         initial_train_data_fraction=0.05, final_model_layer_len=64,
                         active_learning_batch_size=32, training_epochs=1, test_repeat_count=1
                     ),
                         ExperimentConfig(
                             initial_train_data_fraction=0.05, final_model_layer_len=64,
                             active_learning_batch_size=16, training_epochs=1, test_repeat_count=1
                         )
                     ],
                     is_test=True,
                     grayscale=True)
    exp.run_experiments()

def very_quick_badge_test():
    exp = Experiment(name="very_quick_BADGE_test",
                     models=[MiddleFusionModel],
                     query_function_names=["BADGE"],
                     experiment_configs=[ExperimentConfig(
                         initial_train_data_fraction=0.05, final_model_layer_len=64,
                         active_learning_batch_size=32, training_epochs=1, test_repeat_count=1
                     ),
                         ExperimentConfig(
                             initial_train_data_fraction=0.05, final_model_layer_len=64,
                             active_learning_batch_size=16, training_epochs=1, test_repeat_count=1
                         )
                     ],
                     query_function_name_to_extra_options={
                         "BADGE": [KMeansPlusPlusSeeding(), WeightedKMeansSampling()]
                     },
                     is_test=True)
    exp.run_experiments()

def new_experiments(experiment_configs=BASELINE_CONFIGS, grayscale=False):
    """
    These experiments are for evaluating all the active learning methods against each other to see how they fare.
    :return:
    """
    exp = Experiment(
        name="new_experiments",
        models=ALL_MODELS,
        query_function_names=ALL_QUERY_FUNCTION_NAMES,
        experiment_configs=experiment_configs,
        grayscale=grayscale
    )
    exp.run_experiments()

def new_experiments_just_linear(experiment_configs=BASELINE_CONFIGS, grayscale=False):
    """
    These experiments are for evaluating all the active learning methods against each other to see how they fare.
    :return:
    """
    exp = Experiment(
        name="new_experiments",
        models=[MultiModalLateFusionLinearModelInterface],
        query_function_names=ALL_QUERY_FUNCTION_NAMES,
        experiment_configs=experiment_configs,
        grayscale=grayscale
    )
    exp.run_experiments()

def cluster_margin_cluster_methods_experiments(experiment_configs=BASELINE_CONFIGS):
    """
    These experiments are for checking what clustering method helps cluster margin perform the best.
    :return:
    """
    cluster_margin_options = {
        "CLUSTER_MARGIN": [SklearnKMeans(), SklearnAgglomerativeCluster(), SklearnGMM()]
    }
    exp = Experiment(
        name="cluster_margin_cluster_methods_experiments",
        models=ALL_MODELS,
        query_function_names=["CLUSTER_MARGIN"],
        options=cluster_margin_options,
        query_function_name_to_extra_options=BASELINE_CONFIGS,
        experiment_configs=experiment_configs
    )
    exp.run_experiments()

def badge_sample_methods_experiments(experiment_configs=BASELINE_CONFIGS):
    """
    Tehse experiments are to determine what sampling method works best for badge
    :return:
    """
    badge_options = {
        "BADGE": [KMeansPlusPlusSeeding(), WeightedKMeansSampling()]
    }
    exp = Experiment(
        name="badge_sample_methods_experiments",
        models=ALL_MODELS,
        query_function_names=["BADGE"],
        query_function_name_to_extra_options=badge_options,
        experiment_configs=experiment_configs
    )
    exp.run_experiments()

if __name__ == '__main__':
    """
    We'll run all the experiments in this function
    """
    new_experiments(experiment_configs=[other_experiment_config])
#very_quick_test()
    """
    new_experiments_just_linear()
    new_experiments_just_linear(grayscale=True)

    new_experiments(experiment_configs=[other_experiment_config])
    new_experiments(experiment_configs=[other_experiment_config], grayscale=True)

    cluster_margin_cluster_methods_experiments()
    cluster_margin_cluster_methods_experiments(experiment_configs=[other_experiment_config])
    badge_sample_methods_experiments(experiment_configs=[other_experiment_config])
    """
    #very_quick_test()
    #very_quick_test_grayscale()
    #very_quick_badge_test()
    #very_quick_test()
    #new_experiments()
    #cluster_margin_cluster_methods_experiments()
    #badge_sample_methods_experiments()
