from models.resnet import resnet18, resnet34, resnet50
from models.lenet import LeNet
from models.vgg import VGG11, VGG13, VGG16, VGG19


def build_extractor_model(model_name, pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained, model_path='./pretrain_models/resnet/resnet18-f37072fd.pth')
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained, model_path='./pretrain_models/resnet/resnet34-b627a593.pth')
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained, model_path='./pretrain_models/resnet/resnet50-0676ba61.pth')

    elif model_name == 'lenet':
        model = LeNet()

    elif model_name == 'VGG11':
        model = VGG11()
    elif model_name == 'VGG13':
        model = VGG13()
    elif model_name == 'VGG16':
        model = VGG16()
    elif model_name == 'VGG19':
        model = VGG19()

    else:
        model = 'There is no model named ' + model_name

    print(model)
    return model