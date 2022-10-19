import os
from torchvision.io import read_image
import torch

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.dataset_dir = dataset_dir
        class_directories = os.listdir(dataset_dir)
        self.classes = []
        self.length = 0
        for path in class_directories:
            directory_size = len(os.listdir(os.path.join(dataset_dir, path)))
            self.length += directory_size
            self.classes += [(path, directory_size)]


    def __getitem__(self, idx):
        idx_sum = 0
        i = -1
        while idx_sum < idx:
            i += 1
            idx_sum += self.classes[i][1]
        
        if i < 0:
            i = 0
        ## directory name of class
        class_path = self.classes[i][0]
        im_path = os.path.join(self.dataset_dir, class_path)

        ## index of image in class directory = size of class - idx_sum + idx
        directory_index = self.classes[i][1] - (idx_sum - idx) - 1

        
        im_path = os.path.join(im_path, os.listdir(im_path)[directory_index])
        im = read_image(im_path)

        if self.transform:
            im = self.transform(im)
        
        return im, i


    def __len__(self):
        return self.length