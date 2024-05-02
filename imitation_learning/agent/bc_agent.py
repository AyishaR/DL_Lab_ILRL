import torch
import torch.nn as nn
from imitation_learning.agent.networks import *


class BCAgent:

    def __init__(self, history_length=0, lr=0.001, n_classes=4):
        self.net = CNN4(history_length=history_length, n_classes=n_classes).to(non_blocking=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                    lr=lr)
        
    def loss_step(self, X_tensor, y_tensor):
        y_pred = self.net(X_tensor)
        loss = self.loss_fn(y_pred, y_tensor)
        return y_pred, loss
    
    def update(self, X_batch, y_batch):
        X_tensor = torch.tensor(X_batch)
        y_tensor = torch.tensor(y_batch)
        y_pred = self.net(X_tensor)
        loss = self.loss_fn(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del X_tensor
        del y_tensor
        return loss

    def predict(self, X):
        outputs_raw = self.net(torch.tensor(X))
        outputs = torch.argmax(outputs_raw, 1)
        return outputs_raw, outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
