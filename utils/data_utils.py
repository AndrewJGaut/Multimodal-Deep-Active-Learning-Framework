import torch
import numpy as np
from torchvision import datasets, transforms

def get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(
        data_dir="../data/kaggle_satellite_image_classification",
        adjust_contrast=False):
    def my_adjust_contrast():
        def _func(img):
            return transforms.functional.adjust_contrast(img, contrast_factor=0.8)
        return _func

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
                                                      num_workers=0, shuffle=True)
    x = next(iter(all_data_dataloader))[0].numpy()
    y = next(iter(all_data_dataloader))[1].numpy()

    return x, y
