import torch.utils.data as data
from utils import get_data_loader
import numpy as np
import scipy.io
import os


def load_data(data_folder, batch_size, infinite_data_loader, train, num_workers=0, **kwargs):
    data = ASDDataset(root=data_folder)
    data_loader = get_data_loader(data, batch_size=batch_size,
                                  shuffle=True if train else False,
                                  infinite_data_loader=infinite_data_loader,
                                  num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = 2
    return data_loader, n_class


class ASDDataset(data.Dataset):
    def __init__(self, root):
        # 将.mat数据读取至dict：data_mat
        # 分别得到其中的数据train_X和标签train_Y
        data_mat = scipy.io.loadmat(root)
        data_RSFC = data_mat['RSFC']

        sample_len = int(len(data_RSFC))
        train_X = np.array([np.array(data_RSFC[i][0], dtype=np.float32) for i in range(sample_len)])
        train_Y = np.array([0 if data_RSFC[i][2] == 2 else 1 for i in range(sample_len)], dtype=np.int64)

        train_X, train_Y = self.get_sample(train_X, train_Y)

        self.train_X = train_X
        self.train_Y = train_Y

    def __getitem__(self, index):
        train_X, train_Y = self.train_X[index], self.train_Y[index]

        return train_X, train_Y

    def __len__(self):
        return len(self.train_X)

    def get_sample(self, train_X, train_Y):
        train_X = np.expand_dims(train_X, 0).repeat(3, axis=0)
        train_X = np.transpose(train_X, (1, 0, 2, 3))

        return train_X, train_Y


if __name__ == '__main__':
    """
        Parameters
        ----------
        folder_src : path (folder of data)
        batch_size : int
        infinite_data_loader : Epoch-based training (False) / Iteration-based training (True)
        train : train/test
    """

    root = '../data/ASD'
    src = 'LEUVEN'
    data = 'RSFC.mat'
    folder_src = os.path.join(root, src, data)
    source_loader, n_class = load_data(folder_src, batch_size=32,
                                       infinite_data_loader=True,
                                       train=True, num_workers=0)
    print(len(source_loader))
    print(n_class)
    for i, (data, label) in enumerate(source_loader):
        print('第' + str(i) + '次：')
        print(label)
        if i == 10:
            break
