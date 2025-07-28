import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# predict the future 30 days
def predict_future(model, scaler, last_sequence, future_days=30):
    """Predict future prices using the trained model"""
    predictions = []
    current_sequence = last_sequence.clone()
    
    with torch.no_grad():
        for _ in range(future_days):
            # Get prediction (add batch dimension)
            pred = model(current_sequence.unsqueeze(0))
            predictions.append(pred.item())
            
            # Update sequence: remove oldest, add new prediction
            current_sequence = torch.cat([
                current_sequence[1:], 
                pred.reshape(1, 1)
            ])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

def main(csv_path="stock_dataset.csv", model_path="best_lstm_model.pth", future_days=30):
    # Load data
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    # Prepare scaler (must be same as training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df["Close"].values.reshape(-1, 1))
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get last 60 days' sequence
    scaled_data = scaler.transform(df["Close"].values.reshape(-1, 1))
    last_sequence = torch.tensor(scaled_data[-60:], dtype=torch.float32).to(device)
    
    # Make predictions
    future_prices = predict_future(model, scaler, last_sequence, future_days)
    
    # Create future dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
    )
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Historical Price")
    plt.plot(future_dates, future_prices, label="Predicted Price", linestyle="--")
    plt.title(f"Stock Price Prediction (Next {future_days} Days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Return predictions as DataFrame
    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": future_prices.flatten()
    })

if __name__ == "__main__":
    predictions = main(future_days=30)  # Predict next 30 days
    print(predictions)