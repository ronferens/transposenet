from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np


class CameraPoseDatasetWithMetaData(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(CameraPoseDatasetWithMetaData, self).__init__()
        self.img_paths, self.poses, self.seq = read_labels_file(labels_file, dataset_path)
        self.dataset_size = self.poses.shape[0]
        self.transform = data_transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        pose = self.poses[idx]
        seq = self.seq[idx]
        if self.transform:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose, 'seq': seq}
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    seq = df['seq']
    return imgs_paths, poses, seq