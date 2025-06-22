import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels_embed = self.label_embed(labels)
        x = torch.cat([noise, labels_embed], 1)
        return self.model(x).view(-1, 1, 28, 28)

@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

generator = load_generator()

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate 5 Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    images = generator(z, labels).detach().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i].squeeze(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
