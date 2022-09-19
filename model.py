from turtle import forward
import torch
import torch.nn as nn
import timm


class RSNAClassifier(nn.Module):
    def __init__(self, model_name, pretrained=False, checkpoint_path='', 
                 in_chans=3, num_classes=1000, drop_path_rate=0.0):
        """
        Args:
        -----
        model_name: str
            Name of the model to use.
        pretrained: bool
            Whether to load pretrained weights.
        checkpoint_path: str
            Path to model's pretrained weights.
        in_chans: int
            Number of input channels.
        num_classes: int
            Number of output classes.
        drop_path_rate: float
            Drop path rate for the DropPath function.
        """
        super(RSNAClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       checkpoint_path=checkpoint_path,
                                       drop_path_rate=drop_path_rate)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(num_classes=0)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)


if __name__ == '__main__':
    model = RSNAClassifier('convnext_tiny', pretrained=False, num_classes=2, 
                           checkpoint_path='convnext_tiny_22k_1k_384_altered.pth')
    # print(model(torch.randn(32, 3, 224, 224)))
    print(model)