import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import OCRModel

def train_model(dataset, epochs=10, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = OCRModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for (img_data, label) in dataloader:
            x_image, x_type = img_data  # image tensor and one-hot type vector
            optimizer.zero_grad()
            
            outputs = model(x_image, x_type)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f}")
    
    print("✅ Training complete.")
    return model