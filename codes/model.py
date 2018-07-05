from torchvision import models

# Best ResNet models
class Resnet50():
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet50(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)

class Resnet101():

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet101(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)
    
class Resnet152():

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet152(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)

# Best VGG Models
class VGG19_BN():

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.vgg19_bn(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)

class VGG16_BN():
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.vgg16_bn(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)


# Best DenseNet models
class DenseNet121():
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet121(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)

class DenseNet169():
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet169(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)

class DenseNet201():
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet201(pretrained=pretrained)
        self.num_ftrs = model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, no_classes)