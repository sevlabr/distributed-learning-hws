import sys
sys.path.insert(0, '../')

import numpy as np
from pathlib import Path

import torch
import dc_framework


def train_simple_model():
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

    model_dc_framework = dc_framework.init(model, criterion)
    model_dc_framework.train(
        train_data=data_train,
        val_data=data_val
    )
    model_dc_framework.save("tmp.pt")

def test_simple_model(path: Path):
    data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }
    
    trained_model = dc_framework.load(path)
    trained_model.test(test_data=data)

if __name__ == "__main__":
    train_simple_model()
    test_simple_model("tmp.pt")
