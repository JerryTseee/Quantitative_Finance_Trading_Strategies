import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# LSTM is very important in time series analysis such as weather, stock
# 1. Create a simple dataset
# Sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Input: [1, 2, 3] -> Output: 4, [2, 3, 4] -> Output: 5, etc.
def create_dataset(sequence, look_back=3):
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:i + look_back])  # Previous 3 numbers
        y.append(sequence[i + look_back])    # Next number
    return np.array(X), np.array(y)

# Generate sequence
sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X, y = create_dataset(sequence)

# Reshape X to [samples, time steps, features] for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))  # 1 feature per time step

# 2. Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))  # 50 LSTM units, 3 time steps
model.add(Dense(1))  # Output layer for predicting 1 number
model.compile(optimizer='adam', loss='mse')  # Mean squared error loss

# 3. Train the model
model.fit(X, y, epochs=100, verbose=0)  # Train for 100 epochs

# 4. Test the model
test_input = np.array([8, 9, 10])  # Predict the next number after 8, 9, 10
test_input = test_input.reshape((1, 3, 1))  # Reshape for LSTM
prediction = model.predict(test_input, verbose=0)
print(f"Predicted next number: {prediction[0][0]:.2f}")  # Should be close to 11
