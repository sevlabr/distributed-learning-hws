import sys
sys.path.insert(0, '../')

import logging
import numpy as np
from pathlib import Path

import torch
import dc_framework

logger = logging.getLogger("__name__")

def train_simple_model(device: torch.device = torch.device('cpu')):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    criterion = torch.nn.BCELoss()

    data_train = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }
    
    # same data for validation for simplicity
    data_val = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }

    model_dc_framework = dc_framework.init(model, criterion, device)
    model_dc_framework.train(
        train_data=data_train,
        val_data=data_val
    )
    model_dc_framework.save("tmp.pt")

def test_simple_model(path: Path, device: torch.device = torch.device('cpu')):
    data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }
    
    trained_model = dc_framework.load(path, device)
    trained_model.test(test_data=data)

def run_tests(device: torch.device = torch.device('cpu')):
    train_simple_model(device)
    test_simple_model("tmp.pt", device)

if __name__ == "__main__":
    logger.warning('Test on CPU')
    device_cpu = torch.device('cpu')
    run_tests(device_cpu)
    
    if torch.cuda.is_available():
        logger.warning('Test on GPU')
        device_gpu = torch.device('cuda')
        run_tests(device_gpu)
