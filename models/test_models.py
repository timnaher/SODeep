#%% Imports and Parameter Initialization
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_examples = 3000
sequence_length = 1500
sine_wave_length = 500
sine_frequency = 1
noise_std_dev = 0.4

#%% Data Generation
data_sequences = np.zeros((num_examples, sequence_length))
label_sequences = np.zeros((num_examples, sequence_length))

for i in range(num_examples):
    random_start = np.random.randint(0, sequence_length - sine_wave_length)
    sine_wave = 0.5*-np.sin(2 * np.pi * sine_frequency * np.linspace(0, 1, sine_wave_length))
    data_sequence = np.zeros(sequence_length)
    data_sequence[random_start:random_start + sine_wave_length] = sine_wave
    
    # Add white noise
    data_sequence += np.random.normal(0, noise_std_dev, sequence_length)
    
    # Label: 1 for first half, 2 for second half
    label_sequence = np.zeros(sequence_length)
    label_sequence[random_start:random_start + sine_wave_length] = 2
    label_sequence[random_start:random_start + (sine_wave_length // 2)] = 1

    data_sequences[i] = data_sequence
    label_sequences[i] = label_sequence

# Convert to tensors
data_sequences_tensor = torch.tensor(data_sequences, dtype=torch.float32).unsqueeze(1)
label_sequences_tensor = torch.tensor(label_sequences, dtype=torch.long)

# test data generation
test_sequence = np.zeros((num_examples, sequence_length))
for i in range(num_examples):
    random_start = np.random.randint(0, sequence_length - sine_wave_length)
    sine_wave = 0.5*-np.sin(2 * np.pi * sine_frequency * np.linspace(0, 1, sine_wave_length))
    test_sequence[i, random_start:random_start + sine_wave_length] = sine_wave
    test_sequence[i] += np.random.normal(0, noise_std_dev, sequence_length)

test_sequence_tensor = torch.tensor(test_sequence, dtype=torch.float32).unsqueeze(1)


#%% Model Definition
class EEGOscillationDetector(nn.Module):
    def __init__(self, in_channels=1, num_filters=64, kernel_size=3, lstm_hidden_size=128, input_length=1500):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 2
        self.padding = (kernel_size - 1) // 2

        self.causal1 = nn.Conv1d(in_channels, num_filters, 3, stride=self.stride, padding=self.padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.causal2 = nn.Conv1d(num_filters, num_filters, 4, stride=self.stride, padding=self.padding,dilation=2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.causal3 = nn.Conv1d(num_filters, num_filters, 5, stride=self.stride, padding=self.padding,dilation=3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)

        # Downsampled length
        self.downsampled_length = input_length
        for _ in range(2):
            self.downsampled_length = (self.downsampled_length - 1) // self.stride + 1

        # LSTM Decoder
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden_size, batch_first=True)
        self.decoder = nn.Linear(lstm_hidden_size, 3)  # 3 classes

        # Upsampling
        self.upsample = nn.Upsample(size=input_length, mode="linear", align_corners=False)

    def forward(self, x):
        features = self.causal1(x)  # Shape: (batch, num_filters, downsampled_length)
        features = self.relu1(features)
        features = self.dropout1(features)

        features = self.causal2(features)  # Shape: (batch, num_filters, downsampled_length)
        features = self.relu2(features)
        features = self.dropout2(features)

        features = self.causal3(features)  # Shape: (batch, num_filters, downsampled_length)
        features = self.relu3(features)
        features = self.dropout3(features)

        features = features.permute(0, 2, 1)  # Shape: (batch, downsampled_length, num_filters)
        lstm_out, _ = self.lstm(features)  # Shape: (batch, downsampled_length, lstm_hidden_size)
        predictions = self.decoder(lstm_out)  # Shape: (batch, downsampled_length, num_classes)
        predictions = predictions.permute(0, 2, 1)  # Shape: (batch, num_classes, downsampled_length)
        predictions = self.upsample(predictions)  # Shape: (batch, num_classes, input_length)
        return predictions

#%% Model Training
model = EEGOscillationDetector()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 7
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, num_examples, batch_size):
        x_batch = data_sequences_tensor[i:i+batch_size]
        y_batch = label_sequences_tensor[i:i+batch_size]

        logits = model(x_batch)
        #logits = logits.view(-1, 3)
        #y_batch = y_batch.view(-1)

        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i // batch_size) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i//batch_size}/{num_examples//batch_size}], Loss: {loss.item():.4f}')

#%% Testing and Visualization
for example in range(5):
    x_example = test_sequence_tensor[example].squeeze().numpy()
    y_pred = model(test_sequence_tensor[example].unsqueeze(0)).squeeze().argmax(dim=0).detach().numpy()



    plt.figure(figsize=(12, 6))
    plt.plot(x_example, label='Data Sequence (with Noise)', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Prediction Sequence', color='red', alpha=0.7, linestyle='-.')
    plt.title(f'Example Data Sequence with Sine Wave (Example {example+1})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# %%


test_lstm = nn.LSTM(input_size=64,
            hidden_size=16,
            batch_first=True)

x = torch.randn(32, 10, 64) # BTF batch, time, features

output, (h_n, c_n) = test_lstm(x)
print('shape of output:', output.shape)
print('shape of h_n:', h_n.shape)
print('shape of c_n:', c_n.shape)

# %%
