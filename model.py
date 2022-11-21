from turtle import forward
import torch
import torch.nn as nn
import timm


class RSNAClassifier(nn.Module):
    def __init__(self, model_name, pretrained=False, checkpoint_path='', 
                 in_chans=3, num_classes=1000, drop_path_rate=0.0, use_seq_layer=False):
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
        self.rnn = None
        if use_seq_layer:
            self.rnn = nn.LSTM(n_features, n_features, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(n_features * 2, num_classes)
        else:
            self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        if self.rnn is not None:
            batch_size, seqlen, ch, h, w = x.shape # (2, 32, 3, 512, 512)
            x = x.view(-1, ch, h, w) # (64, 3, 512, 512)
            x = self.model(x) # (64, n_features)
            x = x.view(batch_size, seqlen, -1) # (2, 32, n_features)
            out = self.rnn(x)[0] # (2, 32, n_features * 2)
        else:
            out = self.model(x)
        return self.fc(out)


if __name__ == '__main__':
    model = RSNAClassifier('convnext_tiny', pretrained=False, num_classes=2, 
                           checkpoint_path='convnext_tiny_22k_1k_384_altered.pth')
    # print(model(torch.randn(32, 3, 224, 224)))
    print(model)