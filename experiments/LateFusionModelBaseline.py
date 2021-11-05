from models.multimodal.late_fusion_model import LateFusionModel
from models.multimodal.late_fusion_model_with_mean_probability_uncertainty_sampling import LateFusionModelWithMeanProbabilityUncertaintySampling
from models.unimodal.probabilistic_svm import ProbabilisticSVM
from torchvision import datasets, models, transforms
from models.multimodal.combination_functions import MEAN_CLASSIFICATION

PATH_TO_DATA = "../data/kaggle_satellite_image_classification/data"

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(PATH_TO_DATA, transform=transform)


multimodal_model = LateFusionModelWithMeanProbabilityUncertaintySampling([ProbabilisticSVM(), ProbabilisticSVM()], MEAN_CLASSIFICATION)

if __name__ == "__main__":
    pass
    # first, we need to doctor the data so that we have it split evenly into regular and contrasted data
    # then, we just make the multimodal model and watch it go to work!







