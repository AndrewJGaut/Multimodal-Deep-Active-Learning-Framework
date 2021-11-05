import torch
import numpy as np
from torchvision import datasets, transforms

# Detect if we have a GPU available
def get_kaggle_satellite_image_classification_dataset_as_numpy_arrays(
        data_dir = "../data/kaggle_satellite_image_classification",
    ):

    transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    all_data_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), 
                                            num_workers=0, shuffle=False)
    x = next(iter(all_data_dataloader))[0].numpy()
    y = next(iter(all_data_dataloader))[1].numpy()
    
    return x,y

