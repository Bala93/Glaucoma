from torchvision import models
from torch import nn


# Best ResNet models
class Resnet50:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet50(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, no_classes)


class Resnet101:

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet101(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, no_classes)
    
class Resnet152:

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet152(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, no_classes)

# Best VGG Models
# TODO: Handle VGG properly.
class VGG19_BN:

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.vgg19_bn(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.classifier = nn.Linear(self.num_ftrs, no_classes)

class VGG16_BN:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.vgg16_bn(pretrained=pretrained)
        self.num_ftrs = self.model_ft..in_features
        self.model_ft.classifier = nn.Linear(self.num_ftrs, no_classes)


# Best DenseNet models
class DenseNet121:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet121(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.classifier = nn.Linear(self.num_ftrs, no_classes)

class DenseNet169:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet169(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.classifier = nn.Linear(self.num_ftrs, no_classes)

class DenseNet201:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet201(pretrained=pretrained)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.classifier = nn.Linear(self.num_ftrs, no_classes)


class ModelSelect:
    def __init__(self,model_name,pretrained,no_classes):
        self.model_name = model_name
        self.pretrained = pretrained
        self.no_classes = no_classes
    def getModel(self):
        if (self.model_name == 'resnet50'):
            model = Resnet50(self.pretrained,self.no_classes)
        if (self.model_name == 'resnet101'):
            model = Resnet101(self.pretrained,self.no_classes)
        if (self.model_name == 'resnet152'):
            model = Resnet152(self.pretrained,self.no_classes)
        if (self.model_name == 'densenet121'):
            model = DenseNet121(self.pretrained,self.no_classes)
        if (self.model_name == 'densenet169'):
            model = DenseNet169(self.pretrained,self.no_classes)
        if (self.model_name == 'densenet201'):
            model = DenseNet201(self.pretrained,self.no_classes)    
        if (self.model_name == 'vgg16_bn'):
            model = VGG16_BN(self.pretrained,self.no_classes)
        if (self.model_name == 'vgg19_bn'):
            model = VGG19_BN(self.pretrained,self.no_classes)
        return model.model_ft