import logging

import streamlit as st
import tf_keras as keras
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from label_mapping import display_label, is_supported_label

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="ASL Sign Language Classifier",
    page_icon="🤟",
    layout="centered"
)

# 2. Judul dan Deskripsi
st.title("🤟 ASL Sign Language Classifier")
st.write(
    "Unggah gambar bahasa isyarat tangan (American Sign Language) "
    "untuk memperkirakan abjadnya. Skor yang ditampilkan adalah skor output model, bukan probabilitas yang telah dikalibrasi."
)

# 3. Fungsi Memuat Model dengan tf_keras Legacy
@st.cache_resource
def load_asl_model():
    return keras.models.load_model("best_model.h5", compile=False)

try:
    model = load_asl_model()
except Exception:
    LOGGER.exception("Failed to load the Sign-MNIST model.")
    st.error(
        "Model gagal dimuat. Pastikan file `best_model.h5` tersedia dan versi "
        "dependensi sesuai. Detail teknis dicatat di log server."
    )
    st.stop()

# 4. Mapping label mengikuti nilai label asli Sign-MNIST.
# Dataset tidak memiliki kelas J (indeks 9), sehingga indeks itu tidak boleh
# digeser menjadi K. Output indeks 9 ditampilkan sebagai kelas yang tidak didukung.

# 5. Widget Upload Gambar
uploaded_file = st.file_uploader("Unggah gambar isyarat tangan...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("Ukuran gambar maksimal 5 MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file)
        image.load()
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("File tidak dapat dibaca sebagai gambar yang valid.")
        st.stop()

    st.image(image, caption="Gambar Input", width=250)

    with st.spinner("Mengklasifikasikan gambar..."):
        # Pra-pemrosesan gambar mengikuti format input CNN.
        img_gray = image.convert("L").resize((28, 28))
        img_array = np.array(img_gray) / 255.0
        img_array = np.expand_dims(img_array, axis=(-1, 0))  # Shape: (1, 28, 28, 1)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_label = display_label(int(predicted_idx))

    # Display Hasil
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Prediksi Huruf", value=predicted_label)
    
    with col2:
        st.metric(label="Skor output model", value=f"{confidence:.2f}%")
        
    if not is_supported_label(int(predicted_idx)):
        st.warning(
            f"Model memilih output indeks {int(predicted_idx)}, yang tidak memiliki "
            "label huruf statis yang didukung. Hasil ini bukan prediksi huruf yang valid."
        )
    else:
        st.success(
            f"Prediksi model: **Huruf {predicted_label}**. "
            f"Skor output model: **{confidence:.2f}%** (bukan probabilitas terkalibrasi)."
        )