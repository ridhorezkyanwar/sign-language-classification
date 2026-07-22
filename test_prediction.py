import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Muat model yang sudah dilatih
model = tf.keras.models.load_model("best_model.h5")

# 2. Muat gambar dan ubah ke Grayscale / 1 channel (mode 'L')
image_path = "amer_sign2.png"
img = Image.open(image_path).convert('L')

# 3. Ubah ukuran gambar sesuai input model (28x28)
img = img.resize((28, 28))  

# 4. Pra-pemrosesan array gambar
img_array = np.array(img) / 255.0  # Normalisasi nilai piksel (0.0 - 1.0)

# Tambahkan channel dimension: (28, 28) -> (28, 28, 1)
img_array = np.expand_dims(img_array, axis=-1)

# Tambahkan batch dimension: (28, 28, 1) -> (1, 28, 28, 1)
img_array = np.expand_dims(img_array, axis=0)

# 5. Prediksi
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

print("\n" + "="*35)
print(f"Hasil Prediksi Kelas/Kategori: {predicted_class}")
print("="*35)