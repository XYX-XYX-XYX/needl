import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        # 1. Determine which files to load
        if train:
            # List of training files: data_batch_1 through data_batch_5
            file_names = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            # List containing only the test file
            file_names = ['test_batch']
            
        data_list = []
        labels_list = []

        for file_name in file_names:
            file_path = os.path.join(base_folder, file_name)
            raw_data = unpickle(file_path)
            data_list.append(raw_data[b'data'])
            labels_list.extend(raw_data[b'labels'])

        X_flat = np.concatenate(data_list, axis=0)
        self.X = X_flat.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.array(labels_list, dtype=np.uint8)

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        images = self.X[index]
        labels = self.y[index]
        transformation_images = self.apply_transforms(images)

        return transformation_images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
