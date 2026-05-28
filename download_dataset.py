import os
from datasets import load_dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
output_folder = BASE_DIR / "Dataset"
output_file = os.path.join(output_folder, "vietnamese_corpus_10gb.txt")
    
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

TARGET_SIZE_BYTES = 10 * 1024 * 1024 * 1024 

print(f"Bắt đầu tải dữ liệu (Streaming)...")
print(f"Dữ liệu sẽ được ghi trực tiếp vào: {output_file}")
print("Mục tiêu: Đạt ~10.00 GB sẽ tự động dừng.\n")

my_token = os.getenv("HF_TOKEN")
if not my_token:
    raise RuntimeError("Thiếu HF_TOKEN. Hãy đặt biến môi trường HF_TOKEN trước khi chạy loadData.py.")

stream_dataset = load_dataset("uonlp/CulturaX", "vi", split="train", streaming=True, token=my_token)

current_size = 0
line_count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for item in stream_dataset:
        text = item['text'].strip()
        
        if text: 
           
            f.write(text + "\n")
            
            current_size += len(text.encode('utf-8')) + 1 
            line_count += 1
            
            if line_count % 100000 == 0:
                current_gb = current_size / (1024**3)
                print(f"Đã tải {line_count:,} văn bản | Dung lượng hiện tại: {current_gb:.2f} GB")
                
            if current_size >= TARGET_SIZE_BYTES:
                print("\n[Hoàn thành] Đã chạm mốc 10GB. Đang dừng quá trình tải và đóng file!")
                break

final_gb = current_size / (1024**3)
print(f"\nTuyệt vời! File dữ liệu {final_gb:.2f} GB ({line_count:,} văn bản) đã sẵn sàng tại: {output_file}")
