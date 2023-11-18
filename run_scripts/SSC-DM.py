# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy as np
import random

from ..utils.data_loader import Data_Loader
from ..model.lenet import LeNet
from ..model.alexnet import AlexNet
from ..model.resnet import resnet18, resnet50, resnet101

from ..attack.fgsm import FGSM
from ..attack.mim import MIM
from ..attack.pgd import PGD

from ..utils.compute_similarity import compute_euc_dis, compute_manhattan_dis, compute_cos_similarity

def sim_constraint(clean_output, adv_output, dis_metric):
    if dis_metric == "euc":
        clean_output_matrix = compute_euc_dis(clean_output)
        adv_output_matrix = compute_euc_dis(adv_output)
    elif dis_metric == "manhattan":
        clean_output_matrix = compute_manhattan_dis(clean_output)
        adv_output_matrix = compute_manhattan_dis(adv_output)
    elif dis_metric == "cos":
        clean_output_matrix = compute_cos_similarity(clean_output)
        adv_output_matrix = compute_cos_similarity(adv_output)
    
    sim_loss = torch.linalg.norm(clean_output_matrix - adv_output_matrix) / len(clean_output)

    return sim_loss

def train_epoch(model, device, attack_cls, coeff_adv_loss, coeff_sim_loss, dis_metric, train_loader, optimizer, epoch):
    model.train()

    clean_train_loss = 0
    clean_train_correct = 0

    adv_train_loss = 0
    adv_train_correct = 0
    attack_cls.device = device
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        attack_cls.model = model
        adv_data = attack_cls(data, target).to(device)

        optimizer.zero_grad()

        clean_output = model(data)
        adv_output = model(adv_data)

        clean_loss = F.cross_entropy(clean_output, target)
        adv_loss = F.cross_entropy(adv_output, target)

        loss = clean_loss + coeff_adv_loss * adv_loss + coeff_sim_loss * sim_constraint(clean_output, adv_output, dis_metric)
        loss.backward()

        optimizer.step()

        clean_pred = clean_output.argmax(dim=1, keepdim=True)
        adv_pred = adv_output.argmax(dim=1, keepdim=True)

        clean_train_loss += clean_loss.item()
        adv_train_loss += adv_loss.item()

        clean_train_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()
        adv_train_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
    
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        torch.cuda.empty_cache()
    clean_train_loss /= len(train_loader)
    adv_train_loss /= len(train_loader)

    clean_train_acc = clean_train_correct / len(train_loader.dataset)
    adv_train_acc = adv_train_correct / len(train_loader.dataset)
    
    return clean_train_loss, adv_train_loss, clean_train_acc, adv_train_acc

def test_epoch(model, device, attack_cls, coeff_adv_loss, test_loader, epoch):
    model.eval()

    clean_test_loss = 0
    clean_test_correct = 0

    adv_test_loss = 0
    adv_test_correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        attack_cls.model = model
        adv_data = attack_cls(data, target).to(device)
        with torch.no_grad():
            clean_output = model(data)
            adv_output = model(adv_data)

            clean_loss = F.cross_entropy(clean_output, target)
            adv_loss = F.cross_entropy(adv_output, target)

            loss = clean_loss + coeff_adv_loss * adv_loss

            clean_pred = clean_output.argmax(dim=1, keepdim=True)
            adv_pred = adv_output.argmax(dim=1, keepdim=True)

            clean_test_loss += clean_loss.item()
            adv_test_loss += adv_loss.item()

            clean_test_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()
            adv_test_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
        
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))
        torch.cuda.empty_cache()
    clean_test_loss /= len(test_loader)
    adv_test_loss /= len(test_loader)

    clean_test_acc = clean_test_correct / len(test_loader.dataset)
    adv_test_acc = adv_test_correct / len(test_loader.dataset)
    
    return clean_test_loss, adv_test_loss, clean_test_acc, adv_test_acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    model_name = "model_name"
    data_name = "dataset_name"
    lr = 0.01
    momentum = 0.9
    train_batch_size = 128
    test_batch_size = 256
    device = "cuda:0"
    max_iter = 20
    attack_method = "fgsm"
    dis_metric = "euc"
    coeff_adv_loss = 1.0
    coeff_sim_loss = 300.0

    train_data_root = "dataset_path"
    

    train_loader, test_loader = Data_Loader(data_name, train_data_root, train_batch_size, test_batch_size).load_data()
    
    train_net = AlexNet().to(device)
    optimizer = torch.optim.SGD(train_net.parameters(), lr=lr, momentum=momentum, weight_decay=0.001)

    attack_param_set = {
        "fgsm": {"model": train_net, "device": "cuda", "norm": np.inf, "eps": 0.03, "loss": "ce"},
        "pgd": {"model": train_net, "device": "cuda:0", "norm": np.inf, "eps": 0.3, "step_size": 0.0078, "steps": 20, "loss": "ce"},
        "mim": {"model": train_net, "device": "cuda:0", "norm": np.inf, "eps": 0.1, "step_size": 0.0078, "steps": 20, "decay_factor": 0.9, "loss": "ce"}
    }

    if attack_method == "fgsm":
        attack_cls = FGSM(**attack_param_set[attack_method])
    elif attack_method == "pgd":
        attack_cls = PGD(**attack_param_set[attack_method])
    elif attack_method == "mim":
        attack_cls = MIM(**attack_param_set[attack_method])

    for epoch in range(1, max_iter + 1):
        clean_train_loss, adv_train_loss, clean_train_acc, adv_train_acc = train_epoch(train_net, device, attack_cls, coeff_adv_loss, coeff_sim_loss, dis_metric, train_loader, optimizer, epoch)
        clean_test_loss, adv_test_loss, clean_test_acc, adv_test_acc = test_epoch(train_net, device, attack_cls, coeff_adv_loss, test_loader, epoch)

if __name__ == "__main__":
    setup_seed(0)
    main()
