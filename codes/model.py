from torchvision import models
from torch import nn


# ResNet models
class Resnet50:
    
    def __init__(self,pretrained,no_classes):
        # In the network, the last layer is the fc. The FC output should be changed to no_classes,
        # The nn.Linear layer takes in num_features as input and gives no_of_classes as output.
        self.model_ft = models.resnet50(pretrained=pretrained)
        self.model_ft.fc.out_features = no_classes

class Resnet101:

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet101(pretrained=pretrained)
        self.model_ft.fc.out_features = no_classes
    
class Resnet152:

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.resnet152(pretrained=pretrained)
        self.model_ft.fc.out_features = no_classes

# VGG
class VGG16_BN:
    
    def __init__(self,pretrained,no_classes):
        # Updating the final layer out_features by no_classes
        self.model_ft = models.vgg16_bn(pretrained=pretrained)
        self.model_ft.classifier[-1].out_features = no_classes

class VGG19_BN:

    def __init__(self,pretrained,no_classes):
        self.model_ft = models.vgg19_bn(pretrained=pretrained)
        self.model_ft.classifier[-1].out_features = no_classes


# DenseNet models
class DenseNet121:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet121(pretrained=pretrained)
        self.model_ft.classifier.out_features = no_classes

class DenseNet161:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet121(pretrained=pretrained)
        self.model_ft.classifier.out_features = no_classes

class DenseNet169:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet169(pretrained=pretrained)
        self.model_ft.classifier.out_features = no_classes

class DenseNet201:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.densenet201(pretrained=pretrained)
        self.model_ft.classifier.out_features = no_classes

class Inception:
    
    def __init__(self,pretrained,no_classes):
        self.model_ft = models.inception_v3(pretrained=pretrained)
        self.model_ft.fc.out_features = no_classes


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
        # if (self.model_name == 'densenet121'):
        #     model = DenseNet121(self.pretrained,self.no_classes)
        if (self.model_name == 'densenet161'):
            model = DenseNet161(self.pretrained,self.no_classes)
        if (self.model_name == 'densenet169'):
            model = DenseNet169(self.pretrained,self.no_classes)
        if (self.model_name == 'densenet201'):
            model = DenseNet201(self.pretrained,self.no_classes)    
        if (self.model_name == 'vgg16_bn'):
            model = VGG16_BN(self.pretrained,self.no_classes)
        if (self.model_name == 'vgg19_bn'):
            model = VGG19_BN(self.pretrained,self.no_classes)
        if (self.model_name == 'inception'):
            model = Inception(self.pretrained,self.no_classes)

        return model.model_ft
    

if __name__ == "__main__":
    # Available models 
    # Inputs are model_name,is_
    model_names = ['resnet50','resnet101','resnet152','densenet161','densenet169','densenet201','vgg16_bn','vgg19_bn','inception']
    no_class = 2
    for model_name in model_names:
        m = ModelSelect(model_name,True,no_class)
        mm = m.getModel()
        print (mm)
