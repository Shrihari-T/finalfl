import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import get_model
from data import load_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "resnet"  # change to mobilenet / efficientnet


class FLClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        self.model = get_model(MODEL_NAME).to(DEVICE)

        self.train_loader, self.val_loader = load_datasets(3, self.cid)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(2):  # 🔥 IMPORTANT
            for images, labels in self.train_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = self.model(images)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"[Client {self.cid}] Acc:{acc:.4f} Prec:{precision:.4f} Rec:{recall:.4f} F1:{f1:.4f}")

        return 0.0, len(self.val_loader.dataset), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


if __name__ == "__main__":
    cid = sys.argv[1]

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(cid)
    )