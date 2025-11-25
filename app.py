import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Yol Hasar Tespit Sistemi", page_icon="🚧")

st.title("🚧 Yol Hasar Tespit Projesi")
st.markdown(
    """
    <style>
    .stApp {background-color: #f0f2f6;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- MODEL YÜKLEME ---
model_yolu = 'best.pt'

try:
    model = YOLO(model_yolu)
except Exception as e:
    st.error(f"Model yüklenemedi! Hata: {e}")
    st.stop()

# --- KENAR ÇUBUĞU ---
st.sidebar.title("Ayarlar")
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence)", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info("Model: YOLOv8 Nano\nDurum: Hazır")

# --- SEKMELER (RESİM ve VIDEO) ---
tab1, tab2 = st.tabs(["📷 Resim Analizi", "🎥 Video Analizi"])

# --- TAB 1: RESİM İŞLEME ---
with tab1:
    st.header("Fotoğraf Yükle")
    uploaded_file = st.file_uploader("Bir yol fotoğrafı seçin...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Orijinal Resim', use_container_width=True)

        if st.button('Resmi Analiz Et', type="primary"):
            with st.spinner('Analiz yapılıyor...'):
                results = model.predict(image, conf=conf_threshold)
                for result in results:
                    im_array = result.plot()
                    im_output = Image.fromarray(im_array[..., ::-1])

                    with col2:
                        st.image(im_output, caption='Yapay Zeka Sonucu', use_container_width=True)
                        st.success("İşlem Tamam!")

# --- TAB 2: VİDEO İŞLEME ---
with tab2:
    st.header("Video Yükle")
    uploaded_video = st.file_uploader("Bir video dosyası seçin...", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        # 1. Videoyu geçici dosyaya kaydet
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)  # Orijinal videoyu göster

        if st.button('Videoyu Analiz Et ve Hazırla', type="primary"):
            st.warning("Video işleniyor... Bu işlem videonun uzunluğuna göre zaman alabilir.")
            progress_bar = st.progress(0)  # İlerleme çubuğu

            cap = cv2.VideoCapture(tfile.name)

            # Video özelliklerini al (Genişlik, Yükseklik, FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Çıktı Videosu İçin Ayarlar (MP4 Formatı)
            output_path = "islenmis_video.mp4"
            # Codec: 'mp4v' genelde her yerde çalışır
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            st_frame = st.empty()  # Anlık görüntü alanı
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Modeli çalıştır
                results = model.predict(frame, conf=conf_threshold)
                res_plotted = results[0].plot()  # BGR formatında döner (OpenCV için uygun)

                # 1. Dosyaya Yaz (Kaydetme işlemi burada yapılıyor)
                out.write(res_plotted)

                # 2. Ekranda Göster (RGB'ye çevirip Streamlit'e veriyoruz)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, caption=f'İşleniyor: {frame_count}/{total_frames}', use_container_width=True)

                # İlerleme çubuğunu güncelle
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            # İşlem bitince kaynakları serbest bırak
            cap.release()
            out.release()
            progress_bar.empty()

            st.success("Video başarıyla işlendi ve kaydedildi!")

            # --- İNDİRME BUTONU ---
            # Oluşturulan dosyayı oku ve butona ver
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="📥 İşlenmiş Videoyu İndir",
                    data=file,
                    file_name="yol_hasar_tespiti.mp4",
                    mime="video/mp4"
                )