import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import torch
import torch.nn as nn

# load dataset
data = pd.read_csv("spam.csv")

# Remove unnecessary columns
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Rename columns
data = data.rename(columns={'v1': 'label', 'v2': 'message'})

# Vectorization
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(data['message']).toarray()
y = data['label'].values

# Encoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)   # ham=0, spam=1

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

# Build ANN model
class ANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

model = ANN(x_train.shape[1])

# Train model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save pipeline
pipeline = {
    "model_state": model.state_dict(),
    "tfidf": vectorizer,
    "label_encoder": encoder,
    "input_size": x_train.shape[1]
}

joblib.dump(pipeline, "spam_pipeline.pkl")

print("Pipeline saved successfully!")