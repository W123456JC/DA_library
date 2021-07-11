import numpy as np
import scipy.io
import os
import torch
import torch.nn as nn
from visualize.tsne import visualize_tsne
import torch.utils.data as data
from models import build_extractor
import torch.nn.functional as F
import torch.optim as optim
from loss.calculate_loss import cross_entropy


class ASDDataset(data.Dataset):
    def __init__(self, train_X, train_Y, domain_label):
        train_X, train_Y = self.get_sample(train_X, train_Y)

        self.train_X = train_X
        self.train_Y = train_Y
        self.domain_label = domain_label

    def __getitem__(self, index):
        train_X, train_Y = self.train_X[index], self.train_Y[index]
        domain_label = self.domain_label[index]

        return train_X, train_Y, domain_label

    def __len__(self):
        return len(self.train_X)

    def get_sample(self, train_X, train_Y):
        train_X = np.expand_dims(train_X, 0).repeat(3, axis=0)
        train_X = np.transpose(train_X, (1, 0, 2, 3))

        return train_X, train_Y


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    """
        ASD数据集做数据分布可视化
        使用预训练好的resnet提取特征
    """


    ASD = np.load('ASD_data.npz')
    data = ASD['data']
    label = ASD['label']
    domain_label = ASD['domain_label']
    # visualize_tsne(data, domain_label, opentsne=False)
    # visualize_tsne(data, domain_label, opentsne=True)
    # visualize_tsne(data, label, opentsne=False)
    # visualize_tsne(data, label, opentsne=True)

    dataset = ASDDataset(data, label, domain_label)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    extractor = build_extractor.build_extractor_model(model_name='lenet', pretrained=True).cuda()
    classifier = Classifier().cuda()

    opt_extractor = optim.SGD(extractor.parameters(), lr=0.0001, weight_decay=0.0005, momentum=True)
    opt_classifier = optim.SGD(classifier.parameters(), lr=0.0001, weight_decay=0.0005, momentum=True)

    criterion = nn.CrossEntropyLoss().cuda()

    feature_list = []
    label_list = []
    domain_label_list = []
    for epoch in range(30):
        for batch_idx, (data, label, domain_label) in enumerate(dataloader):
            data, label, domain_label = data.cuda(), label.cuda(), domain_label.cuda()

            opt_extractor.zero_grad()
            opt_classifier.zero_grad()

            feature = extractor(data)
            logits = classifier(feature)
            loss = cross_entropy(logits, label, criterion)

            loss.backward()
            opt_extractor.step()
            opt_classifier.step()
            print('epoch:', epoch, ' batch:', batch_idx, ' loss:', loss.item())

            if epoch == 29:
                feature_list.extend(feature.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())
                domain_label_list.extend(domain_label.cpu().detach().numpy())
    feature = np.array(feature_list)
    label = np.array(label_list)
    domain_label = np.array(domain_label_list)
    print(feature.shape)
    print(type(feature))
    print(feature.dtype)
    visualize_tsne(feature, domain_label, opentsne=False)
    visualize_tsne(feature, domain_label, opentsne=True)



