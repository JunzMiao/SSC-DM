# -*- coding: utf-8 -*-

from torchvision import models
 
def resnet18(num_classes):
    net = models.resnet18(num_classes=num_classes)

    return net

def resnet50(num_classes):
    net = models.resnet50(num_classes=num_classes)

    return net

def resnet101(num_classes):
    net = models.resnet101(num_classes=num_classes)

    return net

