import pandas as pd
import numpy as np
import ta
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')

# Preprocessing
data['price'] = data['price'] / 1e9
data['bid_px_00'] = data['bid_px_00'] / 1e9
data['ask_px_00'] = data['ask_px_00'] / 1e9

data['Close'] = data['price']
data['Volume'] = data['size']
data['High'] = data[['bid_px_00', 'ask_px_00']].max(axis=1)
data['Low'] = data[['bid_px_00', 'ask_px_00']].min(axis=1)
data['Open'] = data['Close'].shift(1).fillna(data['Close'])

# Technical Indicators
class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def add_momentum_indicators(self):
        self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close']).rsi()
        macd = ta.trend.MACD(self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_signal'] = macd.macd_signal()
        self.data['MACD_hist'] = macd.macd_diff()
        stoch = ta.momentum.StochasticOscillator(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['Stoch_k'] = stoch.stoch()
        self.data['Stoch_d'] = stoch.stoch_signal()

    def add_volume_indicators(self):
        self.data['OBV'] = ta.volume.OnBalanceVolumeIndicator(self.data['Close'], self.data['Volume']).on_balance_volume()

    def add_volatility_indicators(self):
        bb = ta.volatility.BollingerBands(self.data['Close'])
        self.data['Upper_BB'] = bb.bollinger_hband()
        self.data['Middle_BB'] = bb.bollinger_mavg()
        self.data['Lower_BB'] = bb.bollinger_lband()
        self.data['ATR_1'] = ta.volatility.AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=1).average_true_range()
        self.data['ATR_2'] = ta.volatility.AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=2).average_true_range()
        self.data['ATR_5'] = ta.volatility.AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=5).average_true_range()
        self.data['ATR_10'] = ta.volatility.AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=10).average_true_range()
        self.data['ATR_20'] = ta.volatility.AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=20).average_true_range()

    def add_trend_indicators(self):
        self.data['ADX'] = ta.trend.ADXIndicator(self.data['High'], self.data['Low'], self.data['Close']).adx()
        self.data['+DI'] = ta.trend.ADXIndicator(self.data['High'], self.data['Low'], self.data['Close']).adx_pos()
        self.data['-DI'] = ta.trend.ADXIndicator(self.data['High'], self.data['Low'], self.data['Close']).adx_neg()
        self.data['CCI'] = ta.trend.CCIIndicator(self.data['High'], self.data['Low'], self.data['Close']).cci()

    def add_other_indicators(self):
        self.data['DLR'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['TWAP'] = self.data['Close'].expanding().mean()
        self.data['VWAP'] = (self.data['Volume'] * (self.data['High'] + self.data['Low']) / 2).cumsum() / self.data['Volume'].cumsum()

    def add_all_indicators(self):
        self.add_momentum_indicators()
        self.add_volume_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_other_indicators()
        return self.data

# Create and apply indicators
ti = TechnicalIndicators(data)
df_with_indicators = ti.add_all_indicators().dropna()  # Drop any rows with NaN values
market_features_df = df_with_indicators[35:]  # Adjust based on your requirement

# Preprocess data for transformer input
def preprocess_for_transformer(df):
    # Convert features to tensor-friendly format
    features = df[['Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_k', 'Stoch_d',
                   'OBV', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR_1', 'ADX', '+DI', '-DI', 'CCI']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return torch.tensor(features_scaled, dtype=torch.float32)

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x.squeeze(1)  # Remove sequence dimension

# Initialize model, loss function, and optimizer


input_dim = 17
hidden_dim = 64
num_heads = 4
num_layers = 2
output_dim = 3  # Buy, Sell, Hold

print(input_dim)

model = TransformerModel(input_dim, hidden_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Preprocess data
features = preprocess_for_transformer(market_features_df)
print(features.shape)

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Split data into batches
    for i in range(0, len(features), batch_size):
        batch_features = features[i:i+batch_size]
        batch_targets = torch.randint(0, output_dim, (batch_features.size(0),))  # Example targets
        
        # Forward pass
        outputs = model(batch_features)
        
        # Compute loss
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'transformer_model.pth')

# Load model
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()

# Evaluation
with torch.no_grad():
    test_features = preprocess_for_transformer(market_features_df)  # Replace with actual test data
    predictions = model(test_features)
    recommendations = torch.argmax(predictions, dim=1)

# Print recommendations
print("Generated Recommendations (0: Hold, 1: Buy, 2: Sell):")
print(recommendations)
