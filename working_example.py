#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision import transforms

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from avalanche.logging import TextLogger, InteractiveLogger, TensorboardLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.models import as_multitask
from avalanche.models import SimpleCNN
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR100, CIFAR10
from avalanche.benchmarks import nc_benchmark, benchmark_with_validation_stream

def main():
    num_tasks = 10
    val_size = 0.05
    batch_size = 64
    nepochs = 1
    datadir = 'YOURDATADIR'
    
    
    train_transform = transforms.Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(), 
        transforms.Normalize((0.5071, 0.4866, 0.4409), 
        (0.2009, 0.1984, 0.2023))
    ])
    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), 
        (0.2009, 0.1984, 0.2023))
    ])
    
    cifar_train = CIFAR100(root=datadir, train=True, download=False)
    cifar_test = CIFAR100(root=datadir, train=False, download=False)
    
    # Get data
    scenario_nc = nc_benchmark(cifar_train, cifar_test, num_tasks, 
                               task_labels=True, train_transform=train_transform, 
                               eval_transform=test_transform, 
                               seed=0, class_ids_from_zero_in_each_exp=True)
    scenario = benchmark_with_validation_stream(scenario_nc, 
                                                validation_size=val_size, 
                                                shuffle=True)

    # Create model
    model = SimpleCNN(1)
    model = as_multitask(model, 'classifier')

    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.1, 
                    momentum=0.9, 
                    weight_decay=0.0002)

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]

    evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True),
        loggers=loggers)

    plugins = []

    cl_strategy = Naive(model, optimizer,
                        criterion=CrossEntropyLoss(),
                        train_mb_size=batch_size, 
                        eval_mb_size=batch_size, 
                        device=torch.device('cuda'), 
                        train_epochs=nepochs, 
                        plugins=plugins, 
                        evaluator=evaluator, 
                        eval_every=1)
    
    # TRAINING LOOP
    print('Starting New classes experiment...')
    results = []
    for t, (experience, val_stream) in enumerate(zip(scenario.train_stream, 
                                                 scenario.valid_stream)):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience, eval_streams=[val_stream], num_workers=4)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(scenario.test_stream[:t+1]))

if __name__ == '__main__':
    main()
