import gradio as gr
from PIL import Image
import os
import pandas as pd
import uuid

def process_and_generate_csv(files):
    saved_dir = "saved_images"
    os.makedirs(saved_dir, exist_ok=True)

    data = []

    # Nếu chỉ có 1 ảnh, Gradio không trả list
    if not isinstance(files, list):
        files = [files]

    for idx, file in enumerate(files):
        img = Image.open(file)
        width, height = img.size
        filename = f"image_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(saved_dir, filename)
        img.save(save_path)

        data.append({
            "Tên ảnh": filename,
            "Chiều rộng": width,
            "Chiều cao": height,
            "Đường dẫn": save_path
        })

    # Ghi vào file CSV
    csv_path = "output.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return csv_path  # trả về file csv để người dùng tải về

# Giao diện Gradio
demo = gr.Interface(
    fn=process_and_generate_csv,
    inputs=gr.File(file_types=["image"], file_count="multiple", label="Tải lên ảnh"),
    outputs=gr.File(label="Tải xuống file CSV kết quả"),
    title="Xử lý ảnh và Xuất CSV",
    description="Tải lên ảnh và nhận lại file CSV chứa thông tin từng ảnh"
)

demo.launch(share=True)
