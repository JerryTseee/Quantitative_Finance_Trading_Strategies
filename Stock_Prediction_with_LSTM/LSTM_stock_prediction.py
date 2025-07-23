def main(csv_path="stock_dataset.csv"):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error


    # Load Dataset
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Close"])
    plt.title("Real Stock Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    # Prepare sequences
    def prepare_data(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step])
            y.append(data[i + time_step])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = prepare_data(scaled_price, time_step)

    # Split train/test
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # LSTM Model
    class StockLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2):
            super(StockLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop with best model checkpointing
    epochs = 1000
    batch_size = 64
    best_mse = float("inf")
    train_loss = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        total_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = perm[i:i+batch_size]
            batch_X, batch_y = X_train[indices].to(device), y_train[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss.append(total_loss)
        
        # Every 100 epochs: evaluate and checkpoint if better
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test.to(device)).cpu().numpy()
            test_pred_unscaled = scaler.inverse_transform(test_pred)
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1).numpy())
            mse = mean_squared_error(y_test_unscaled, test_pred_unscaled)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss:.6f}, Test MSE: {mse:.4f}")

            if mse < best_mse:
                best_mse = mse
                torch.save(model.state_dict(), "best_lstm_model.pth")
                print(f"✅ New best model saved at epoch {epoch+1} with MSE: {mse:.4f}")

    # Final test evaluation
    model.load_state_dict(torch.load("best_lstm_model.pth"))
    model.eval()
    with torch.no_grad():
        final_pred = model(X_test.to(device)).cpu().numpy()

    final_pred_unscaled = scaler.inverse_transform(final_pred)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1).numpy())

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, label="Real Price")
    plt.plot(final_pred_unscaled, label="Best Model Prediction")
    plt.title("Stock Price Prediction - Best LSTM Model")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Final MSE
    final_mse = mean_squared_error(y_test_unscaled, final_pred_unscaled)
    print(f"✅ Best Model Final Test MSE: {final_mse:.4f}")

if __name__ == "__main__":
    main()