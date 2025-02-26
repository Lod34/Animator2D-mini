# app_gradio.py

import torch
import torch.nn as nn
import gradio as gr
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np

# --------------------------------------------------
# Definizione della stessa architettura usata in training
# --------------------------------------------------
class SimpleGenerator(nn.Module):
    def __init__(self, text_embedding_dim, img_size=64, channels=3):
        super(SimpleGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 8 * 8 * 128),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, text_embedding):
        x = self.fc(text_embedding)
        x = x.view(-1, 128, 8, 8)
        x = self.conv(x)
        return x

# --------------------------------------------------
# Imposta il dispositivo (GPU se disponibile)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Inizializza il modello e carica i pesi salvati
# --------------------------------------------------
model = SimpleGenerator(text_embedding_dim=512).to(device)
model.load_state_dict(torch.load("Animator2D-mini.pth", map_location=device))
model.eval()

# --------------------------------------------------
# Inizializza tokenizer e text encoder
# --------------------------------------------------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# --------------------------------------------------
# Funzione per generare lo sprite a partire dagli input utente
# --------------------------------------------------
def generate_sprite(description, num_frames, action, direction):
    # Combina gli input in un'unica stringa di prompt
    prompt = f"{description}, frames: {num_frames}, action: {action}, view: {direction}"
    
    # Tokenizza il prompt
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    # Ottieni l'embedding testuale (utilizziamo il token [CLS])
    with torch.no_grad():
        text_embedding = text_encoder(**tokens).last_hidden_state[:, 0, :]
    
    # Genera l'immagine
    with torch.no_grad():
        output = model(text_embedding)
    
    # Post-processing: da tensore (valori in [-1, 1]) a immagine in formato PIL
    img_tensor = output[0].cpu().detach()
    img_tensor = (img_tensor + 1) / 2  # Scala i valori a [0,1]
    img_np = img_tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # Converte da CHW a HWC
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    return img

# --------------------------------------------------
# Creazione dell'interfaccia Gradio
# --------------------------------------------------
iface = gr.Interface(
    fn=generate_sprite,
    inputs=[
        gr.Textbox(label="Character Description", value="A brave knight"),
        gr.Number(label="Number of Animation Frames", value=1),
        gr.Textbox(label="Character Action", value="sword attack"),
        gr.Textbox(label="Viewing Direction", value="side")
    ],
    outputs=gr.Image(label="Animated Sprite"),
    title="Sprite Generator",
    description="Genera uno sprite animato a partire da una descrizione, numero di frame, azione e direzione di visualizzazione."
)

# Avvia l'interfaccia web
iface.launch()
