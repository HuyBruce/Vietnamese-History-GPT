from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import MiniGPT, block_size, device

print("="*50)
print(" HỆ THỐNG SINH VĂN BẢN (TEXT GENERATION) - MINI GPT")
print("="*50)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

FILE_TOKENIZER = DATASET_DIR / "vi_tokenizer.json"
FILE_SAVE_MODEL = BASE_DIR / "mini_gpt_vietnamese.pt"

print("-> Đang nạp từ điển (Tokenizer)...")
tokenizer = Tokenizer.from_file(str(FILE_TOKENIZER))

print("-> Đang nạp não bộ (Model Weights)...")
model = MiniGPT().to(device)

checkpoint = torch.load(FILE_SAVE_MODEL, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("-> Đã nạp xong! Sẵn sàng chém gió.\n")

def generate_text(prompt, max_new_tokens=50, temperature=0.8):
    """
    prompt: Câu mồi (Ví dụ: "Trần Hưng Đạo là")
    max_new_tokens: Số từ muốn máy viết thêm
    temperature: Độ sáng tạo. (0.1 là an toàn/rập khuôn, 1.0 là ngẫu nhiên/sáng tạo)
    """
    encoded = tokenizer.encode(prompt).ids
    idx = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] 
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    while True:
        print("-" * 50)
        user_prompt = input("Nhập câu mồi (hoặc gõ 'exit' để thoát): ")
        if user_prompt.lower() == 'exit':
            break
            
        print("\n[Mini-GPT đang suy nghĩ...]")
        result = generate_text(user_prompt, max_new_tokens=100, temperature=0.7)
        
        print("\nKết quả:")
        print(f">> {result}")
        print("\n")