import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from pathlib import Path

class VietnameseGPTDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, block_size=256, max_samples=100000):
        print("Đang nạp Tokenizer...")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.block_size = block_size
        self.data = []
        
        print("Đang mã hóa văn bản thành mảng số (Tokenization)...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text_chunk = ""
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                text_chunk += line.strip() + " "
        
        encoded = self.tokenizer.encode(text_chunk).ids
        
        for i in range(0, len(encoded) - block_size):
            self.data.append(encoded[i : i + block_size + 1])
            
        print(f"Đã tạo xong Dataset với {len(self.data)} sequences.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR / "Dataset"

    file_path = DATASET_DIR / "vietnamese_corpus_10gb.txt"
    tokenizer_path = DATASET_DIR / "vi_tokenizer.json"
    
    dataset = VietnameseGPTDataset(file_path, tokenizer_path, block_size=256, max_samples=50000)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    x_batch, y_batch = next(iter(dataloader))
    print(f"Kích thước X: {x_batch.shape}") 
    print(f"Kích thước Y: {y_batch.shape}") 