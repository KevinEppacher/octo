import jax.numpy as jnp
import numpy as np
from transformers import CLIPVisionModel, CLIPFeatureExtractor
import flax.linen as nn
import torch

class CLIPWrapper(nn.Module):
    model_name: str = "openai/clip-vit-base-patch16"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        # Lade das CLIP-Vision-Modell
        clip_vision_encoder = CLIPVisionModel.from_pretrained(self.model_name)
        feature_extractor = CLIPFeatureExtractor.from_pretrained(self.model_name)

        # ⬇️ Bilder auf 224x224 resizen (CLIP braucht genau diese Größe)
        pixel_values = feature_extractor(images=observations, return_tensors="pt")["pixel_values"]

        # Feature Extraction
        outputs = clip_vision_encoder(pixel_values=pixel_values)

        # Rückgabe der Token-Embeddings
        return outputs.last_hidden_state

# 🔥 **Testfunktion für CLIPWrapper**
def test_clip_wrapper():
    # 🖼️ 1️⃣ Dummy-Bild erzeugen (1 Bild, 256x256, 3 Kanäle)
    dummy_image = np.random.rand(256, 256, 3) * 255  # RGB-Werte [0, 255]
    dummy_image = dummy_image.astype(np.uint8)  # In Ganzzahlen umwandeln

    # ⬆️ 2️⃣ Bild in eine Liste packen (wie ein Batch mit 1 Bild)
    observations = [dummy_image]

    # 🏗️ 3️⃣ Modell instanziieren und ausführen
    model = CLIPWrapper()
    outputs = model.apply({}, observations)  # Forward-Pass

    # 🔍 4️⃣ Ergebnisse überprüfen
    assert isinstance(outputs, torch.Tensor), "❌ Output ist kein Torch-Tensor!"
    print(f"✅ Output Shape: {outputs.shape}")  # Erwartete Form: (1, 197, 768)

# 🚀 **Starte den Test**
if __name__ == "__main__":
    test_clip_wrapper()


# import numpy as np
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

# # Lade das CLIP-Modell und den Prozessor
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# # Funktion, um ein Bild zu laden und in das richtige Format zu bringen
# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return image

# # Beispielpfad für ein Bild (ersetze diesen mit deinem Bildpfad)
# example_image_path = "/home/kevin/Documents/Master/3_Semester/Modern_Robot_Concepts/octo/examples/apple.png"  # Stelle sicher, dass dieses Bild existiert
# image = load_and_preprocess_image(example_image_path)

# # Verwende den Prozessor, um das Bild in das Modell einzuspeisen
# inputs = processor(images=image, return_tensors="pt", padding=True)

# # Extrahiere die Feature-Repräsentationen des Bildes
# image_features = clip_model.get_image_features(**inputs)

# # Normalisiere die Features für weitere Tests
# normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)

# # Ausgabe der Dimensionen und Beispielwerte
# print("Feature Shape:", normalized_features.shape)
# print("Feature Values (erste 5):", normalized_features[0, :5].detach().numpy())

# # Simuliere, wie die Features in eine Token-Liste konvertiert werden könnten
# tokens = normalized_features.detach().numpy()
# print("Simulated Tokens:", tokens)
