import torch
import numpy as np

from src.models.resnet1d import ResNet1D
from src.models.labels import MODULATION_CLASSES


MODEL_PATH = "./outputs/models/best_resnet_model.pth"


def load_model():
    model = ResNet1D(num_classes=len(MODULATION_CLASSES))

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict(model, iq_sample):
    """
    Predict modulation class from one IQ sample.

    Expected input:
        iq_sample: numpy array of shape (1024, 2)

    Returns:
        predicted_class
        confidence
        probabilities
    """

    iq = np.asarray(iq_sample, dtype=np.float32)

    if iq.shape != (1024, 2):
        raise ValueError(
            f"Expected IQ sample shape (1024, 2), got {iq.shape}"
        )

    # Convert (1024, 2) -> (2, 1024)
    iq = iq.T

    # Add batch dimension -> (1, 2, 1024)
    tensor = torch.from_numpy(iq).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())

    predicted_class = MODULATION_CLASSES[predicted_index]

    return predicted_class, confidence, probabilities[0].numpy()
