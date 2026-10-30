import torch
import torch.nn as nn


class FallLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1]).squeeze(-1)


def load_lstm_model(model_path, device):
    model = FallLSTM().to(device)

    state = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model