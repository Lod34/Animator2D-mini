# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
import torchvision.transforms as T

# -----------------------------
# Parametri di addestramento
# -----------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-4
IMG_SIZE = 64  # Riduciamo la dimensione per semplicità

# -----------------------------------
# Carica il dataset da Hugging Face
# -----------------------------------
dataset = load_dataset("pawkanarek/spraix_1024", split="train")

# --------------------------------------------------
# Trasformazioni per le immagini:
# - Ridimensionamento a IMG_SIZE x IMG_SIZE
# - Conversione in Tensor
# - Normalizzazione a valori tra -1 e 1
# --------------------------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Lambda(lambda x: x.convert('RGB')),  # Converti da RGBA a RGB
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --------------------------------------------------
# Inizializza tokenizer e text encoder (CLIP)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # Sposta il modello su GPU

# --------------------------------------------------
# Definizione di una rete generativa semplice
# Questa rete prende in input l'embedding testuale e genera un'immagine.
# --------------------------------------------------
class SimpleGenerator(nn.Module):
    def __init__(self, text_embedding_dim, img_size=IMG_SIZE, channels=3):
        super(SimpleGenerator, self).__init__()
        # Strato Fully Connected per mappare l'embedding testuale a una mappa di feature
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 8 * 8 * 128),
            nn.ReLU()
        )
        # Rete di convoluzioni trasposte per generare l'immagine finale (64x64)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),  # Output: 64x64
            nn.Tanh()  # Output normalizzato tra -1 e 1
        )
    
    def forward(self, text_embedding):
        x = self.fc(text_embedding)
        x = x.view(-1, 128, 8, 8)  # Rimodelliamo il vettore in una mappa 2D
        x = self.conv(x)
        return x

# --------------------------------------------------
# Imposta il dispositivo (GPU se disponibile)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGenerator(text_embedding_dim=512).to(device)  # CLIP produce embedding di dimensione 512
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()  # Per semplicità usiamo l'MSE come funzione di loss

# --------------------------------------------------
# Funzione di collate per il DataLoader
# --------------------------------------------------
def collate_fn(batch):
    images = []
    texts = []
    for item in batch:
        # Assumiamo che il dataset contenga le seguenti colonne; se non presenti, si usano valori di default.
        description = item.get("description", "sprite")
        frames = str(item.get("frames", "1"))
        action = item.get("action", "idle")
        direction = item.get("direction", "front")
        # Combiniamo gli input in un'unica stringa di prompt
        full_prompt = f"{description}, frames: {frames}, action: {action}, view: {direction}"
        texts.append(full_prompt)
        # Applica le trasformazioni all'immagine
        img = item["image"]
        img = transform(img)
        images.append(img)
    images = torch.stack(images)
    return {"texts": texts, "images": images}

# Crea il DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# --------------------------------------------------
# Ciclo di addestramento
# --------------------------------------------------
model.train()
for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        texts = batch["texts"]
        images = batch["images"].to(device)
        
        # Tokenizza i prompt testuali e sposta su GPU
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        tokens = {key: val.to(device) for key, val in tokens.items()}  # Sposta i token su GPU
        
        # Ottieni gli embedding testuali
        with torch.no_grad():
            text_embeddings = text_encoder(**tokens).last_hidden_state[:, 0, :]
        
        # Passaggio in avanti: genera l'immagine
        outputs = model(text_embeddings)
        loss = criterion(outputs, images)
        
        # Backward e aggiornamento dei pesi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# Salva il modello allenato
torch.save(model.state_dict(), "Animator2D-mini.pth")
print("Modello salvato come Animator2D-mini.pth")
