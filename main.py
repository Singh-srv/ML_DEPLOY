from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np
import os
from typing import List

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoders using relative paths
label_encoder = joblib.load(os.path.join(BASE_DIR,  "LST_label_encoder.pkl"))
char_to_index = joblib.load(os.path.join(BASE_DIR, "LST_char_to_index_LSTM.pkl" ))

# Load the model architecture
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert to embeddings
        out, _ = self.lstm(x)  # LSTM forward
        out = self.fc(out[:, -1, :])  # Take last timestep output
        return out

# Initialize model
vocab_size = len(char_to_index)
embedding_dim = 16
hidden_size = 128
num_layers = 2
num_classes = len(label_encoder.classes_)

model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, num_layers, num_classes)

# Load model weights
model_path = os.path.join(BASE_DIR, "lstm_char_embedding.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
model.eval()

# FastAPI app
app = FastAPI()

# Function to encode ID numbers
MAX_LENGTH = 20  # Fixed length
def encode_id_number(id_number, max_length=MAX_LENGTH):
    id_number = str(id_number)[:max_length]  # Trim if too long
    encoded = [char_to_index.get(c, 0) for c in id_number]  # Convert to indices
    return encoded + [0] * (max_length - len(encoded))  # Pad with 0s

# Request model for a single ID prediction
class Item(BaseModel):
    id_number: str  # Expecting an ID number as a string

# Request model for multiple ID predictions
class MultipleItems(BaseModel):
    id_numbers: List[str]  # Expecting a list of ID numbers

# Prediction endpoint for a single ID
@app.post("/predict/")
def predict_category(item: Item):
    encoded_sample = encode_id_number(item.id_number)
    encoded_sample = np.array(encoded_sample).reshape(1, MAX_LENGTH)
    encoded_sample = torch.tensor(encoded_sample, dtype=torch.long)

    with torch.no_grad():
        output = model(encoded_sample)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Thresholding for low confidence
    threshold = 0.9
    if confidence.item() < threshold:
        return {"category": "Unknown (Low Confidence)"}

    predicted_category = label_encoder.inverse_transform([predicted_class.item()])[0]
    return {"category": predicted_category}

# Prediction endpoint for multiple IDs
@app.post("/predict_multiple/")
def predict_multiple_categories(items: MultipleItems):
    results = []
    for id_number in items.id_numbers:
        encoded_sample = encode_id_number(id_number)
        encoded_sample = np.array(encoded_sample).reshape(1, MAX_LENGTH)
        encoded_sample = torch.tensor(encoded_sample, dtype=torch.long)

        with torch.no_grad():
            output = model(encoded_sample)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Thresholding for low confidence
        threshold = 0.9
        if confidence.item() < threshold:
            results.append({"id_number": id_number, "category": "Unknown (Low Confidence)"})
        else:
            predicted_category = label_encoder.inverse_transform([predicted_class.item()])[0]
            results.append({"id_number": id_number, "category": predicted_category})

    return {"predictions": results}
