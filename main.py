from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI()

# Assuming you have loaded your model here
model = torch.load('path_to_your_model.pth')  # Load your trained model
model.eval()

def preprocess_image(image):
    # Convert image to tensor, normalize, resize or any other preprocessing your model requires
    image = torch.Tensor(image).unsqueeze(0)  # Example preprocessing
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    image = preprocess_image(image)  # Apply necessary preprocessing
    with torch.no_grad():
        prediction = model(image)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
