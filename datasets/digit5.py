import torch.utils.data as data
import numpy as np
from scipy.io import loadmat
from PIL import Image
from utils import dense_to_one_hot
from utils import get_data_loader
import torchvision.transforms as transforms


base_dir = '../data/digit5'


def load_data(domain_name, batch_size, infinite_data_loader, num_workers=0, scale=32, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_image, train_label, test_image, test_label = return_dataset(domain_name)
    train_dataset = digit5Dataset(train_image, train_label, transform)
    test_dataset = digit5Dataset(test_image, test_label, transform)

    train_loader = get_data_loader(train_dataset, batch_size=batch_size,
                                   shuffle=True,
                                   infinite_data_loader=infinite_data_loader,
                                   num_workers=num_workers, **kwargs, drop_last=True)
    test_loader = get_data_loader(test_dataset, batch_size=batch_size,
                                  shuffle=False,
                                  infinite_data_loader=infinite_data_loader,
                                  num_workers=num_workers, **kwargs, drop_last=False)
    n_class = 10

    return train_loader, test_loader, n_class


def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    elif data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist()
        # print(train_image.shape)
    elif data == 'mnistm':
        train_image, train_label, \
        test_image, test_label = load_mnistm()
        # print(train_image.shape)
    elif data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps()
    elif data == 'syn':
        train_image, train_label, \
        test_image, test_label = load_syn()

    else:
        train_image, train_label, \
        test_image, test_label = None, None, None, None
        print('There is no digit dataset named ' + data)

    return train_image, train_label, test_image, test_label


class digit5Dataset(data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.train_X = X
        self.train_Y = Y
        self.transform = transform

    def __getitem__(self, index):
        train_X, train_Y = self.train_X[index], self.train_Y[index]

        if train_X.shape[0] != 1:
            train_X = Image.fromarray(np.uint8(np.asarray(train_X.transpose((1, 2, 0)))))

        elif train_X.shape[0] == 1:
            im = np.uint8(np.asarray(train_X))
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            train_X = Image.fromarray(im)

        if self.transform is not None:
            train_X = self.transform(train_X)

        return train_X, train_Y

    def __len__(self):
        return len(self.train_X)


def load_mnist(scale=True, usps=False, all_use=False):
    mnist_data = loadmat(base_dir + '/mnist_data.mat')
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test = mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:25000]
    test_label = test_label[:25000]
    print('mnist train X shape->', mnist_train.shape)
    print('mnist train y shape->', train_label.shape)
    print('mnist test X shape->', mnist_test.shape)
    print('mnist test y shape->', test_label.shape)

    return mnist_train, train_label, mnist_test, test_label


def load_mnistm(scale=True, usps=False, all_use=False):
    mnistm_data = loadmat(base_dir + '/mnistm_with_label.mat')
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']

    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)

    mnistm_train = mnistm_train[:25000]
    train_label = train_label[:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    print('mnist_m train X shape->', mnistm_train.shape)
    print('mnist_m train y shape->', train_label.shape)
    print('mnist_m test X shape->', mnistm_test.shape)
    print('mnist_m test y shape->', test_label.shape)
    return mnistm_train, train_label, mnistm_test, test_label


def load_svhn():
    svhn_train = loadmat(base_dir + '/svhn_train_32x32.mat')
    svhn_test = loadmat(base_dir + '/svhn_test_32x32.mat')
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)

    print('svhn train y shape before dense_to_one_hot->', svhn_train['y'].shape)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    print('svhn train y shape after dense_to_one_hot->',svhn_label.shape)
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])
    svhn_train_im = svhn_train_im[:25000]
    svhn_label = svhn_label[:25000]
    svhn_test_im = svhn_test_im[:9000]
    svhn_label_test = svhn_label_test[:9000]
    print('svhn train X shape->',  svhn_train_im.shape)
    print('svhn train y shape->',  svhn_label.shape)
    print('svhn test X shape->',  svhn_test_im.shape)
    print('svhn test y shape->', svhn_label_test.shape)

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test


def load_usps(all_use=False):
    dataset = loadmat(base_dir + '/usps_28x28.mat')
    data_set = dataset['dataset']
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = label_train[inds]

    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))

    # img_test = dense_to_one_hot(img_test)
    label_train = dense_to_one_hot(label_train)
    label_test = dense_to_one_hot(label_test)

    img_train = np.concatenate([img_train, img_train, img_train, img_train], 0)
    label_train = np.concatenate([label_train, label_train, label_train, label_train], 0)

    print('usps train X shape->', img_train.shape)
    print('usps train y shape->', label_train.shape)
    print('usps test X shape->', img_test.shape)
    print('usps test y shape->', label_test.shape)

    return img_train, label_train, img_test, label_test


def load_syn(scale=True, usps=False, all_use=False):
    syn_data = loadmat(base_dir + '/syn_number.mat')
    syn_train = syn_data['train_data']
    syn_test =  syn_data['test_data']
    syn_train = syn_train.transpose(0, 3, 1, 2).astype(np.float32)
    syn_test = syn_test.transpose(0, 3, 1, 2).astype(np.float32)
    syn_labels_train = syn_data['train_label']
    syn_labels_test = syn_data['test_label']

    train_label = syn_labels_train
    inds = np.random.permutation(syn_train.shape[0])
    syn_train = syn_train[inds]
    train_label = train_label[inds]
    test_label = syn_labels_test

    train_label = dense_to_one_hot(train_label)
    test_label = dense_to_one_hot(test_label)

    print('syn number train X shape->',  syn_train.shape)
    print('syn number train y shape->',  train_label.shape)
    print('syn number test X shape->',  syn_test.shape)
    print('syn number test y shape->', test_label.shape)
    return syn_train, train_label, syn_test, test_label


if __name__ == '__main__':
    s1_loader, s2_loader, n_class = load_data('mnistm', 32)
    print(s1_loader)
    for i, (imgs, labels) in enumerate(s1_loader):
        print(imgs.shape)
        print(labels)
        break
