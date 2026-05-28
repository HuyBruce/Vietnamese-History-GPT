from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
file_path = DATASET_DIR / "vietnamese_corpus_10gb.txt"

num_lines = 10

if file_path.exists():
    print(f"--- HIỂN THỊ {num_lines} DÒNG ĐẦU TIÊN ---")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i in range(num_lines):
            line = f.readline()
            
            if not line:
                break
                
            print(f"[{i + 1}] {line.strip()}\n")
            
    print("--- HOÀN TẤT ---")
else:
    print(f"Lỗi: Không tìm thấy file tại {file_path}")