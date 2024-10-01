import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import jaccard_score
from unetpp import model_unetpp  # Import the U-Net++ model
from dataloader import get_dataloaders  # Import the function to get dataloaders

# Dice Score Function
def dice_score(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

# Model Training Function
def train_model(model, dataloaders, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, masks in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloaders['train'])}")

        # Validation Phase
        model.eval()
        dice = 0.0
        with torch.no_grad():
            for inputs, masks in dataloaders['val']:
                outputs = model(inputs)
                dice += dice_score(masks, outputs)
        print(f"Epoch {epoch+1}/{num_epochs}, DICE Score: {dice/len(dataloaders['val'])}")

if __name__ == "__main__":
    # Load the dataloaders (no arguments)
    dataloaders = get_dataloaders()

    # Train the U-Net++ model
    train_model(model_unetpp, dataloaders, num_epochs=10)
