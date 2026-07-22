import streamlit as st
import tf_keras as keras
import numpy as np
from PIL import Image

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
    "untuk memprediksi abjadnya secara *real-time*!"
)

# 3. Fungsi Memuat Model dengan tf_keras Legacy
@st.cache_resource
def load_asl_model():
    return keras.models.load_model("best_model.h5", compile=False)

try:
    model = load_asl_model()
except Exception as e:
    st.error(f"Gagal memuat model `best_model.h5`. Error: {e}")
    st.stop()

# 4. Mapping Label (ASL MNIST Standard)
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# 5. Widget Upload Gambar
uploaded_file = st.file_uploader("Unggah gambar isyarat tangan...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Input", width=250)
    
    with st.spinner("Mengklasifikasikan gambar..."):
        # Pra-pemrosesan Gambar
        img_gray = image.convert('L').resize((28, 28))
        img_array = np.array(img_gray) / 255.0
        img_array = np.expand_dims(img_array, axis=(-1, 0))  # Shape: (1, 28, 28, 1)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_label = LABELS.get(predicted_idx, f"Kelas {predicted_idx}")

    # Display Hasil
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Prediksi Huruf", value=predicted_label)
    
    with col2:
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        
    st.success(f"Model memprediksi gambar ini sebagai **Huruf {predicted_label}** dengan tingkat kepastian **{confidence:.2f}%**.")