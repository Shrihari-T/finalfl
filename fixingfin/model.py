import torch.nn as nn
import torchvision.models as models

def get_model(model_name="resnet"):

    if model_name == "resnet":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    else:
        raise ValueError("Invalid model name")

    return model