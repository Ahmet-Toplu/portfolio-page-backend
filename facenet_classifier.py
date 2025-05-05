import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

class FaceNetClassifier(nn.Module):
    def __init__(self, num_emotions=8, freeze_facenet=True):
        super(FaceNetClassifier, self).__init__()
        # 1) Load FaceNet
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        
        if freeze_facenet:
            for param in self.facenet.parameters():
                param.requires_grad = False
        
        # 2) Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),    # first dense
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),    # second dense
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_emotions)  # final logits
        )

    def forward(self, x):
        # If facenet is in eval mode and freeze_facenet=True, it uses no_grad().
        # For partial fine-tuning, we set freeze_facenet=False & call train() on it 
        # or just manually set requires_grad to True for certain layers.
        with torch.no_grad() if not self.facenet.training else torch.enable_grad():
            embeddings = self.facenet(x)  # shape: (batch_size, 512)
        out = self.classifier(embeddings)
        return out
    
# a simple emotion mapping—adjust order/names to match your training
EMOTION_MAP = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral", "other"
]

# Preprocessing pipeline must mirror what you used during training
_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_model(weights_path="facenet_ec_0.7543.pth", device="cpu"):
    model = FaceNetClassifier(num_emotions=len(EMOTION_MAP), freeze_facenet=True)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict_emotion(image, model, device="cpu"):
    """
    image: a PIL.Image in RGB mode
    model: a loaded FaceNetClassifier
    returns: string label
    """
    img_t = _transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        idx = logits.argmax(dim=1).item()
    return EMOTION_MAP[idx]