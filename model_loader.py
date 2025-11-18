import timm
import torch.nn as nn

def load_embedding_model(model_name="mobilenetv3_small_100", weights_path=None):
    model = timm.create_model(model_name, pretrained=True)
    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model.eval()