import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Dataset and plot first
df = pd.read_csv("stock_dataset.csv")
print(df.head())
plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Close"])
plt.title("Real Stock Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()


# Scale Closing Prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_price = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
print(scaled_price[:5])


def prepare_data(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

time_step = 60
X, y = prepare_data(scaled_price, time_step)
X = torch.FloatTensor(X).reshape(-1, time_step, 1).to(device)


# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, time_step):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, time_step * 10)
        self.lstm = nn.LSTM(input_size=10, hidden_size=50, batch_first=True) # using LSTM as generator
        self.fc3 = nn.Linear(50, 1)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, time_step, 10)
        x, _ = self.lstm(x)
        x = torch.clamp(torch.relu(self.fc3(x)), 0.0, 1.0) # relu as activation function
        return x
    
# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, time_step):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_step, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, time_step)
        return self.model(x)
    

# initialize models
latent_dim = 100
generator = Generator(latent_dim, time_step).to(device)
discriminator = Discriminator(time_step).to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002) # optimizer for generator
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002) # optimizer for discriminator


# GAN Training Function
def train_gan(epochs, batch_size = 32):
    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            real_seq = X[i:i+batch_size].to(device)
            batch_size_actual = real_seq.shape[0]

            # real labels
            real_labels = torch.ones((batch_size_actual, 1), device=device)
            fake_labels = torch.zeros((batch_size_actual, 1), device=device)

            # Train Discriminator
            z = torch.randn((batch_size_actual, latent_dim), device=device)
            fake_seq = generator(z)
            d_loss_real = criterion(discriminator(real_seq), real_labels)
            d_loss_fake = criterion(discriminator(fake_seq.detach()), fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()


            # Train Generator
            z = torch.randn((batch_size_actual, latent_dim), device=device)
            fake_seq = generator(z)
            g_loss = criterion(discriminator(fake_seq), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")



# Start Training
train_gan(epochs=1000)

# Generate Visualization of Generated Stock Prices
z = torch.randn((1, latent_dim), device=device)
generated = generator(z).detach().cpu().numpy().reshape(-1, 1)
generated = scaler.inverse_transform(generated)

real = scaler.inverse_transform(X[0].numpy().reshape(-1, 1))
plt.figure(figsize=(14, 7))
plt.plot(real, label="Real")
plt.plot(generated, linestyle="--", label="Generated")
plt.title("Real vs Generated Stock Prices")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()

# MSE
mse = mean_squared_error(real, generated)
print(f"Mean Squared Error: {mse:.4f}")

# Save Models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")