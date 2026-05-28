from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

file_path = DATASET_DIR / "vietnamese_corpus_10gb.txt"
output_model = DATASET_DIR / "vi_tokenizer.json"


print("Đang khởi tạo cấu trúc Tokenizer...")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ],
    show_progress=True
)


def get_training_corpus(
    file_path,
    batch_size=10000,
    max_lines=5000000
):
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []

        for i, line in enumerate(f):

            if i >= max_lines:
                break

            batch.append(line.strip())

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


print("Bắt đầu cho Tokenizer học từ vựng...")

tokenizer.train_from_iterator(
    get_training_corpus(file_path),
    trainer=trainer
)

tokenizer.save(str(output_model))

print(f"Hoàn tất! Tokenizer đã được lưu tại: {output_model}")

