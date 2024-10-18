import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform, device):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(self.device)
        return img


def get_dataloaders(data_paths, transforms, device, batch_size, train_perc=90, seed=0):
    """
    Splits the dataset into training and testing partitions, and returns DataLoaders for each partition.
    Args:
        data_paths (list): List of paths to data.
        transforms (callable): Transformations to be applied to the images.
        batch_size (int): Number of samples per batch to load.
        train_perc (int, optional): Percentage of data to be used for training. Defaults to 90.
        seed (int, optional): Random seed for shuffling the data. Defaults to 0.
    Returns:
        tuple: A tuple containing:
            - dataloader_train (DataLoader): DataLoader for the training partition.
            - dataloader_test (DataLoader): DataLoader for the testing partition.
            - idxs_train (ndarray): Array of indices for the training partition.
            - idxs_test (ndarray): Array of indices for the testing partition.
    """

    num_samples = len(data_paths)
    np.random.seed(seed)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(num_samples / 100 * train_perc))
    
    # Shuffle all stimulus images
    idxs = np.arange(num_samples)
    np.random.shuffle(idxs)

    # Assign (train_perc) % of the shuffled stimulus images to the training partition,
    # and (100- train_perc) % to the test partition
    idxs_train, idxs_test = idxs[:num_train], idxs[num_train:]

    # Get dataloaders
    dataloader_train = \
        DataLoader(ImageDataset(data_paths, idxs_train, transforms, device),
                   batch_size=batch_size)
    dataloader_test = \
        DataLoader(ImageDataset(data_paths, idxs_test, transforms, device),
                   batch_size=batch_size)
    
    print('DataLoaders created successfully! (Shuffule=False for DataLoader)')
    return dataloader_train, dataloader_test, idxs_train, idxs_test