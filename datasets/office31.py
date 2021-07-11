from torchvision import datasets, transforms
from utils import get_data_loader
import os


def load_data(data_folder, batch_size, infinite_data_loader, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_data_loader(data, batch_size=batch_size,
                                  shuffle=True if train else False,
                                  infinite_data_loader=infinite_data_loader,
                                  num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class


if __name__ == '__main__':
    """
        Parameters
        ----------
        folder_src : path (folder of data)
        batch_size : int
        infinite_data_loader : Epoch-based training (False) / Iteration-based training (True)
        train : train/test
    """

    root = '../data/office31'
    src = 'dslr'
    folder_src = os.path.join(root, src)
    source_loader, n_class = load_data(folder_src, batch_size=32,
                                       infinite_data_loader=True,
                                       train=True, num_workers=0)
    print(len(source_loader))
    print(n_class)
