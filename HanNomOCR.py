import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import os

# --- Cấu hình đường dẫn Model ---
MODEL_DIR = './HanNom_det/'
MODEL_NAME = 'HanNom_v1.pdmodel'
PARAMS_NAME = 'HanNom_v1.pdiparams'

# Cấu hình giao diện

st.set_page_config(page_title="Hán OCR", page_icon="🏯")
st.title("Hệ thống nhận diện chữ Hán-Nôm cổ")
st.markdown(
    """
    <style>
    /* 1. Background toàn trang */
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/old-paper.png");
        background-color: #f4e4bc;
        background-attachment: fixed;
    }

    /* 2. Tùy chỉnh ô File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #ede0c4; /* Màu sáng hơn background một chút */
        border: 2px dashed #8b5a2b; /* Viền nét đứt màu nâu gỗ */
        border-radius: 15px;
        padding: 10px;
    }

    /* 3. Chỉnh màu chữ bên trong ô upload */
    [data-testid="stFileUploader"] section {
        background-color: #fdf5e6; /* Màu kem nhạt bên trong */
        color: #4a342e;
        border-radius: 10px;
    }
    
    /* 4. Chỉnh màu nút "Browse files" */
    button[kind="secondary"] {
        background-color: #8b5a2b !important;
        color: white !important;
        border: none !important;
    }
    button[kind="secondary"]:hover {
        background-color: #4a342e !important;
        border: none !important;
    }

    /* 5. Chỉnh màu tiêu đề và chữ chung */
    h1, h2, h3, p, span, label {
        color: #4a342e !important;
        font-family: 'Georgia', serif;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("Tải ảnh văn bản cổ lên để hệ thống tự động tìm kiếm các dòng chữ.")
st.markdown("Có khả năng nhận diện tốt với các văn bản chữ Hán được viết hoặc in theo chiều dọc từ trên xuống dưới, phải sang trái.")

# Load mô hình (Sử dụng cache để không phải load lại mỗi khi nhấn nút)
@st.cache_resource
def load_model():
    # Kiểm tra file trước khi load để tránh tự động tải model mặc định
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
        st.error(f"❌ Không tìm thấy {MODEL_NAME} trong thư mục {MODEL_DIR}")
        return None
        
    return PaddleOCR(
        det_model_dir=MODEL_DIR,
        det_model_filename=MODEL_NAME,
        det_params_filename=PARAMS_NAME,
        use_angle_cls=False,
        use_gpu=False,
        lang='en'
    )

ocr = load_model()

display_name = MODEL_NAME.replace('.pdmodel', '')

if ocr is not None:
    st.markdown(
    f"""
    <div style="
        background-color: #d4e0b5; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #bbc99a;
        margin-bottom: 20px;
        text-align: center;">
        <span style="
            color: #10f70c; 
            font-family: 'Georgia', serif; 
            font-size: 1.2rem; 
            font-weight: bold;">
            Đang nhận diện chữ Hán-Nôm cổ với 
            <span style="font-family: 'Georgia', serif; font-style: bold;">
                {display_name}
            </span>
        </span>
    </div>
    """, 
    unsafe_allow_html=True
)

# Trình tải ảnh
uploaded_file = st.file_uploader("Chọn ảnh Hán-Nôm...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    with st.spinner('Đang phân tích dữ liệu...'):
        # Chạy nhận diện (chỉ dùng Detection)
        result = ocr.ocr(img_array, rec=False)
    
    # Hiển thị kết quả
    st.subheader("Kết quả phân tích:")
    
    # Lấy các tọa độ box
    boxes = [line for line in result[0]]
    
    # Vẽ box lên ảnh
    res_img = draw_ocr(img_array, boxes)
    
    # Hiển thị ảnh kết quả
    st.image(res_img, caption=f"Tìm thấy {len(boxes)} vùng văn bản", use_column_width=True)
    
    st.success(f"Hoàn thành! Đã phát hiện {len(boxes)} dòng chữ.")


st.markdown(
    """
    <hr style="border:1px solid #8b5a2b; opacity: 0.3;">
    <div style="text-align: center; color: #4a342e; font-family: 'Georgia', serif; padding: 20px;">
        <p style="margin-bottom: 5px;">© 2026 Dự án Số hóa Di sản Hán-Nôm</p>
        <p style="font-size: 0.8rem; font-style: italic; opacity: 0.8;">
            Đây chỉ là phiên bản thử nghiệm. Có thể xảy ra sai sót trong quá trình nhận diện.
        </p>
        <p style="font-size: 0.8rem; font-style: italic; opacity: 0.8;">
            Rất mong nhận được sự đóng góp ý kiến từ cộng đồng để cải thiện chất lượng hệ thống. Cảm ơn bạn !
        </p>
        <p style="font-size: 0.8rem; font-style: italic; opacity: 1;">
            Liên hệ tại email: pmtrung1504@gmail.com
        </p>
        <p style="font-size: 1.2rem; margin-top: 10px;">📜 🏛️ 🏮</p>
    </div>
    """,
    unsafe_allow_html=True
)