"""Linear models are NOT working right now."""


from models.multimodal.late_fusion_model import LateFusionModel
from models.multimodal.late_fusion_model_with_mean_probability_uncertainty_sampling import LateFusionModelWithMeanProbabilityUncertaintySampling
from models.unimodal.probabilistic_svm import ProbabilisticSVM
from torchvision import datasets, models, transforms
from models.multimodal.combination_functions import MEAN_CLASSIFICATION
from test_framework.tester import Tester
import numpy as np
import torch


def my_adjust_contrast():
    def _func(img):
        return transforms.functional.adjust_contrast(img, contrast_factor=0.8)

    return _func

def get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(
        data_dir = "../data/kaggle_satellite_image_classification",
        adjust_contrast=False
    ):


    transform = None
    if adjust_contrast:
        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        my_adjust_contrast()])
    else:
        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    all_data_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                            num_workers=0, shuffle=False)
    x = next(iter(all_data_dataloader))[0].numpy()
    y = next(iter(all_data_dataloader))[1].numpy()

    return x,y



PATH_TO_DATA = "../data/kaggle_satellite_image_classification/data"
"""
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(PATH_TO_DATA, transform=transform)

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



second_modality_transform =transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     my_adjust_contrast()])

first_modality = datasets.ImageFolder(PATH_TO_DATA, transform=transform)
second_modality =  datasets.ImageFolder(PATH_TO_DATA, transform=second_modality_transform)
"""
first_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA)
second_modality = get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(PATH_TO_DATA, True)

tester_x = np.stack((first_modality[0], second_modality[0]), axis=1) #np.stack((first_modality[0], second_modality[0]))
tester_y = first_modality[1]

multimodal_model = LateFusionModelWithMeanProbabilityUncertaintySampling([ProbabilisticSVM(), ProbabilisticSVM()], MEAN_CLASSIFICATION)

if __name__ == "__main__":
    tester = Tester(tester_x, tester_y, training_epochs=1, active_learning_loop_count=2)
    tester.test_model(multimodal_model)
    tester.plot_results()







