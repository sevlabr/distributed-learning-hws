import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module):
    return DCFramework(model, criterion)

def load(path: Path):
    model = DCFramework(
        torch.nn.Linear(2, 1), # very silly
        torch.nn.BCELoss()
    )
    
    model.load(path)
    
    return model

class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion

    def forward(self, feature, target):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss
        }

    def train(
        self,
        train_data: Dict[str, np.array],
        val_data: Dict[str, np.array],
        batch_size: int = 1,
        n_epochs: int = 10
    ):
        logger.warning("Loading data...")
        train_data, val_data = Dataset(train_data), Dataset(val_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        val_dataloader = val_data.get_dataloader(batch_size=batch_size)
        
        logger.warning("Starting training...")
        
        accuracy = 0
        for n_epoch in range(n_epochs):
            if (n_epoch + 1) % 5 == 0:
                logger.warning(f"Epoch: {n_epoch + 1}; Accuracy: {accuracy}")
            
            self.model.train()
            for batch in train_dataloader:
                output = self.forward(*batch)
                loss = output["loss"]
                loss.backward()
                self.optimizer.step()
            
            accuracy = self._test_model(val_dataloader)
            
        logger.warning("Training is finished!\n")

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)
    
    def _test_model(self, test_dataloader):
        self.model.eval()
        
        correct = 0
        
        for batch in test_dataloader:
            with torch.no_grad():
                feature, target = batch
                output = self.model(feature)
                output = (output > 0.5).float()
                correct += (output == target).float().sum()
            
        accuracy = 100 * correct / len(test_dataloader)

        return accuracy
    
    def test(self, test_data: Dict[str, np.array], batch_size: int = 1):
        logger.warning("Testing model...")
        test_data = Dataset(test_data)
        test_dataloader = test_data.get_dataloader(batch_size=batch_size)
        accuracy = self._test_model(test_dataloader)
        
        logger.warning(f"Accuracy = {accuracy} %\n")
        
    def load(self, path: Path):
        state = torch.load(path)
        
        self.model.state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
    