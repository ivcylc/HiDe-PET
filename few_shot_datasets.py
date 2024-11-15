import os

import os.path
import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree

import numpy as np
import random
import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image
from continual_datasets.continual_datasets import Imagenet_R, CUB200, StanfordCars
from datasets import get_dataset
from collections import defaultdict
from torch.utils.data import DataLoader
import time

class EpisodeSampler(object):
    def __init__(self, data_source, class_ids, num_episodes, num_ways, num_shots) -> None:
        self.data_source = data_source
        self.class_ids = class_ids
        self.num_episodes = num_episodes
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.samples = defaultdict(list)
        data_loader = DataLoader(self.data_source, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=100)
        start_time = time.time()
        for i, (_, label) in enumerate(data_loader):
            if label in class_ids:
                self.samples[label.item()].append(i)
        print(f"sample time: {time.time() - start_time}")

    def __iter__(self):
        for _ in range(self.num_episodes):
            episode_classes = random.sample(self.class_ids, k=self.num_ways)
            episode_samples = []
            
            for c in episode_classes:
                selected_samples = random.sample(self.samples[c], self.num_shots)
                episode_samples.extend(selected_samples)
            yield episode_samples
    
    def __len__(self):
        return self.num_episodes
    

class QuerySampler(object):
    def __init__(self, data_source, class_ids, num_queries) -> None:
        self.data_source = data_source
        self.class_ids = class_ids
        self.num_queries = num_queries
        
    def __iter__(self):
        validation_sampels = []
        for c in self.class_ids:
            samples = [i for i, (_, lable) in enumerate(self.data_source) if lable == c]
            selected_samples = random.sample(samples, k=self.num_queries)
            validation_sampels.extend(selected_samples) 
        yield validation_sampels

    def __len__(self):
        return 1
    

class EpisodeDataset:
    def __init__(self, data, episode_samples=None):
        self.episode_samples = episode_samples
        self.data = data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target where target is index of the target class.
        """
        img, target = self.data[self.episode_samples[index]]
        return img, target
    
    def __len__(self):
        return len(self.episode_samples)
    

class QueryDataset:
    def __init__(self, data, query_samples=None):
        self.query_samples = query_samples
        self.data = data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target where target is index of the target class.
        """
        img, target = self.data[self.query_samples[index]]
        return img, target
    
    def __len__(self):
        return len(self.query_samples)
       


def get_episode_dataset(dataset, episode_samples, data_source=None):
    if dataset in ['Imagenet-R', 'CUB200', 'Cars196', 'CIFAR100', 'Dogs', 'Flowers102', 'Aircraft']:
        return EpisodeDataset(data_source, episode_samples)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    

def get_query_dataset(dataset, query_samples, data_source=None):
    if dataset in ['Imagenet-R', 'CUB200', 'Cars196', 'CIFAR100', 'Dogs', 'Flowers102', 'Aircraft']:
        return QueryDataset(data_source, query_samples)
    else:
        raise  ValueError('Unknown dataset: {}'.format(dataset))
    
def get_full_dataset(dataset, full_samples, data_source=None):
    if dataset in ['Imagenet-R', 'CUB200', 'Cars196', 'CIFAR100', 'Dogs', 'Flowers102', 'Aircraft']:
        return FullDataset(data_source, full_samples)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


class FullSampler(object):
    def __init__(self, data_source, class_ids, num_ways) -> None:
        self.data_source = data_source
        self.class_ids = class_ids
        self.num_ways = num_ways
        self.samples = defaultdict(list)
        data_loader = DataLoader(self.data_source, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=100)
        start_time = time.time()
        for i, (_, label) in enumerate(data_loader):
            if label in class_ids:
                self.samples[label.item()].append(i)
        print(f"sample time: {time.time() - start_time}")

    def __iter__(self):
        for _ in range(1):
            episode_classes = random.sample(self.class_ids, k=self.num_ways)
            episode_samples = []
            
            for c in episode_classes:
                selected_samples = random.sample(self.samples[c], len(self.samples[c]))
                episode_samples.extend(selected_samples)
            yield episode_samples
    
    def __len__(self):
        return 1
    

class FullDataset:
    def __init__(self, data, full_samples=None):
        self.full_samples = full_samples
        self.data = data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target where target is index of the target class.
        """
        img, target = self.data[self.full_samples[index]]
        return img, target
    
    def __len__(self):
        return len(self.full_samples)