import torch, os
from PIL import Image

class galaxy_Dataset(torch.utils.data.Dataset):

    def __init__(self, root, transform= None, size: float = 1.0):
        self.root = root
        self.transform = transform
        self.fns = os.listdir(self.root)
        self.size = size

    def __len__(self):
        return int(len(self.fns) * self.size)

    def __getitem__(self, index):
        fn = self.fns[index]
        img_path = os.path.join(self.root, fn)
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img
    
class CelebA(torch.utils.data.Dataset):

    def __init__(self, root, transform= None, size: float = 1.0):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(self.root, 'img_align_celeba')
        self.fns = os.listdir(self.img_dir)
        self.size = size

    def __len__(self):
        return int(len(self.fns) * self.size)

    def __getitem__(self, index):
        fn = self.fns[index]
        img_path = os.path.join(self.img_dir, fn)
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img
    
