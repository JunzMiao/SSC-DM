# -*- coding: UTF-8 -*-

from torch import nn

def criterion_adv(loss_name, outputs, labels, target_labels):
    
    if loss_name=="ce":
        loss = nn.CrossEntropyLoss()
        if target_labels:
            criterion = -loss(outputs, target_labels)
        else:
            criterion = loss(outputs, labels)

    return criterion