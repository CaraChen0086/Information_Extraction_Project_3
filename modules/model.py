import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        feat_dim=15,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, input_features, lengths=None):
        """
        input_features: (B, T, D)
        lengths: optional valid frame counts for packed LSTM computation
        """
        if lengths is None:
            lstm_out, _ = self.lstm(input_features)
        else:
            lengths = lengths.detach().to("cpu").clamp(min=1)
            packed = pack_padded_sequence(
                input_features,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=input_features.size(1),
            )

        x = self.norm(lstm_out)
        x = self.dropout(x)
        return self.classifier(x)
