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
with gr.Blocks() as demo:
    gr.Markdown("# Sprite Generator")
    gr.Markdown("Generate animated sprites from text descriptions")
    
    with gr.Row():
        with gr.Column():
            description = gr.Text(label="Character Description", value="pixel art character")
            frames = gr.Number(label="Number of Animation Frames", value=1, minimum=1)
            action = gr.Text(label="Character Action", value="idle")
            direction = gr.Text(label="Viewing Direction", value="front")
            generate_btn = gr.Button("Generate Sprite")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Sprite", type="pil")
    
    generate_btn.click(
        fn=generate_sprite,
        inputs=[description, frames, action, direction],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
