import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tokenizers import Tokenizer

vocab_size = 30000
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
seed = 1337

batch_size = 16
learning_rate = 3e-4
min_learning_rate = 3e-5
warmup_steps = 1000
max_epochs = 1
steps_per_epoch = None  # None = gan dung 1 luot qua train tokens; dat so cu the de gioi han thoi gian.
grad_clip = 1.0
weight_decay = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_lines = None  # None = dung toan bo corpus; vi du 500000 de test nhanh.
val_fraction = 0.02
rebuild_token_cache = False
tokenize_batch_lines = 2000

log_interval = 100
eval_interval = 1000
eval_iters = 50
save_interval = 5000
preview_interval = 5000
preview_max_new_tokens = 80

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

FILE_DATA = DATASET_DIR / "vietnamese_corpus_10gb.txt"
FILE_TOKENIZER = DATASET_DIR / "vi_tokenizer.json"
FILE_TOKEN_CACHE = DATASET_DIR / "vietnamese_tokens.uint16.bin"
FILE_TOKEN_META = DATASET_DIR / "vietnamese_tokens.meta.json"
FILE_SAVE_MODEL = BASE_DIR / "mini_gpt_vietnamese.pt"
FILE_BEST_MODEL = BASE_DIR / "mini_gpt_vietnamese_best.pt"

CHECKPOINT_VERSION = 2
TOKEN_DTYPE = np.uint16

PREVIEW_PROMPTS = [
    "Việt Nam là",
    "Hôm nay trời",
    "Theo thông tin mới nhất",
]



def set_seed():
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def file_signature(path):
    return {
        "size": os.path.getsize(path),
        "mtime": int(os.path.getmtime(path)),
    }


def expected_cache_meta():
    return {
        "source": file_signature(FILE_DATA),
        "tokenizer": file_signature(FILE_TOKENIZER),
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "max_lines": max_lines,
        "separator_token": "[EOS]",
    }


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_matches(meta, expected):
    if not meta:
        return False
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return True


def build_token_cache(expected):
    print("-> Đang xây token cache từ corpus. Lần đầu sẽ mất khá lâu, các lần sau sẽ nạp nhanh hơn.")
    tokenizer = Tokenizer.from_file(str(FILE_TOKENIZER))
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != vocab_size:
        raise ValueError(f"Tokenizer vocab_size={actual_vocab_size}, nhưng code đang đặt vocab_size={vocab_size}.")

    eos_id = tokenizer.token_to_id("[EOS]")
    separator_ids = [eos_id] if eos_id is not None else []
    dtype_limit = np.iinfo(TOKEN_DTYPE).max

    total_lines = 0
    total_tokens = 0
    batch_lines = []

    def flush_batch(out_file):
        nonlocal total_tokens, batch_lines
        if not batch_lines:
            return

        encodings = tokenizer.encode_batch(batch_lines)
        ids = []
        for enc in encodings:
            if enc.ids:
                ids.extend(enc.ids)
                ids.extend(separator_ids)

        if ids:
            max_id = max(ids)
            if max_id > dtype_limit:
                raise ValueError(f"Token id {max_id} vượt quá giới hạn {TOKEN_DTYPE}.")
            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(out_file)
            total_tokens += len(ids)

        batch_lines = []

    os.makedirs(os.path.dirname(FILE_TOKEN_CACHE), exist_ok=True)
    with open(FILE_DATA, "r", encoding="utf-8") as src, open(FILE_TOKEN_CACHE, "wb") as out:
        for line in src:
            if max_lines is not None and total_lines >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            batch_lines.append(line)
            total_lines += 1

            if len(batch_lines) >= tokenize_batch_lines:
                flush_batch(out)

            if total_lines % 100000 == 0:
                print(f"   Đã xử lý {total_lines:,} dòng | {total_tokens:,} tokens")

        flush_batch(out)

    meta = {
        **expected,
        "total_lines": total_lines,
        "total_tokens": total_tokens,
    }
    with open(FILE_TOKEN_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"-> Token cache hoàn tất: {total_tokens:,} tokens từ {total_lines:,} dòng.")
    return meta


def ensure_token_cache():
    expected = expected_cache_meta()
    meta = load_json(FILE_TOKEN_META)

    if (
        not rebuild_token_cache
        and os.path.exists(FILE_TOKEN_CACHE)
        and cache_matches(meta, expected)
    ):
        print(f"-> Dùng token cache có sẵn: {FILE_TOKEN_CACHE}")
        print(f"-> Tổng tokens trong cache: {meta['total_tokens']:,}")
        return meta

    return build_token_cache(expected)


class TokenCorpus:
    def __init__(self, token_cache_path, val_fraction, block_size):
        self.data = np.memmap(token_cache_path, dtype=TOKEN_DTYPE, mode="r")
        total_tokens = len(self.data)
        min_tokens = 2 * (block_size + 1)
        if total_tokens < min_tokens:
            raise ValueError(f"Corpus chỉ có {total_tokens:,} tokens, cần tối thiểu {min_tokens:,}.")

        val_tokens = max(block_size + 1, int(total_tokens * val_fraction))
        val_tokens = min(val_tokens, total_tokens - block_size - 1)
        train_tokens = total_tokens - val_tokens

        self.train_data = self.data[:train_tokens]
        self.val_data = self.data[train_tokens:]

    def get_batch(self, split, batch_size, block_size, generator, device):
        data = self.train_data if split == "train" else self.val_data
        high = len(data) - block_size
        if high <= 0:
            raise ValueError(f"Tập {split} quá ngắn cho block_size={block_size}.")

        starts = torch.randint(high, (batch_size,), generator=generator).tolist()
        x_np = np.stack([data[i:i + block_size] for i in starts]).astype(np.int64)
        y_np = np.stack([data[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)

        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)
        return x, y


# 3. KIEN TRUC MO HINH MINI-GPT

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > block_size:
            raise ValueError(f"Độ dài input {T} vượt quá block_size={block_size}")

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_view = logits.reshape(B * T, C)
            targets_view = targets.reshape(B * T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss


# 4. TRAINING HELPERS: LR SCHEDULE, EVAL, CHECKPOINT, PREVIEW
def get_lr(global_step, total_steps):
    if warmup_steps > 0 and global_step < warmup_steps:
        return learning_rate * (global_step + 1) / warmup_steps

    if global_step >= total_steps:
        return min_learning_rate

    decay_steps = max(1, total_steps - warmup_steps)
    decay_ratio = min(1.0, (global_step - warmup_steps) / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


@torch.no_grad()
def estimate_loss(model, corpus, use_amp, global_step):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        eval_generator = torch.Generator()
        eval_generator.manual_seed(seed + global_step * 17 + (0 if split == "train" else 1))

        for k in range(eval_iters):
            x, y = corpus.get_batch(split, batch_size, block_size, eval_generator, device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        bos_id = tokenizer.token_to_id("[BOS]")
        ids = [bos_id if bos_id is not None else 0]

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return tokenizer.decode(idx[0].tolist())


def print_preview(model, tokenizer):
    print("\n[PREVIEW SINH VĂN BẢN]")
    for prompt in PREVIEW_PROMPTS:
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=preview_max_new_tokens,
            temperature=0.8,
            top_k=50,
        )
        print(f"- Prompt: {prompt}")
        print(f"  {text[:500]}")
    print()


def save_checkpoint(model, optimizer, scaler, global_step, best_val_loss, path, train_generator):
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "model_config": {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
        },
        "train_config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": warmup_steps,
            "max_epochs": max_epochs,
            "steps_per_epoch": steps_per_epoch,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    }, path)
    print(f"\n[LƯU TRỮ] Đã lưu checkpoint step {global_step:,}: {path}")


def _cpu_rng_state(state, name):
    if state is None:
        return None
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.uint8)
    if state.dtype != torch.uint8:
        state = state.to(dtype=torch.uint8)
    state = state.detach().cpu()
    if state.dim() != 1:
        raise ValueError(f"{name} không phải RNG state hợp lệ.")
    return state


def load_checkpoint_if_exists(model, optimizer, scaler, train_generator):
    if not os.path.exists(FILE_SAVE_MODEL):
        print("\n-> Không tìm thấy checkpoint cũ. Sẽ train mới từ đầu.")
        return 0, float("inf")

    print(f"\n[!] Tìm thấy checkpoint: {FILE_SAVE_MODEL}")
    checkpoint = torch.load(FILE_SAVE_MODEL, map_location="cpu", weights_only=False)

    if checkpoint.get("checkpoint_version", 1) < CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint cũ không tương thích với bản train mới. "
            "Hãy xóa mini_gpt_vietnamese.pt rồi train lại từ đầu."
        )
    if checkpoint.get("vocab_size", vocab_size) != vocab_size:
        raise ValueError("Checkpoint không khớp vocab_size với code hiện tại.")
    if checkpoint.get("block_size", block_size) != block_size:
        raise ValueError("Checkpoint không khớp block_size với code hiện tại.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    train_generator.set_state(_cpu_rng_state(checkpoint["train_generator_state"], "train_generator_state"))

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(_cpu_rng_state(checkpoint["torch_rng_state"], "torch_rng_state"))
    if device == 'cuda' and checkpoint.get("cuda_rng_state_all") is not None:
        cuda_rng_state_all = [
            _cpu_rng_state(state, f"cuda_rng_state_all[{idx}]")
            for idx, state in enumerate(checkpoint["cuda_rng_state_all"])
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    print(f"-> Resume từ global step {global_step:,}. Best val loss hiện tại: {best_val_loss:.4f}")
    return global_step, best_val_loss


if __name__ == "__main__":
    print("=" * 50)
    print(" KHỞI ĐỘNG TRAIN BASE MODEL MINI-GPT TIẾNG VIỆT")
    print("=" * 50)
    print(f"-> Thiết bị: {device.upper()}")
    set_seed()

    ensure_token_cache()
    tokenizer = Tokenizer.from_file(str(FILE_TOKENIZER))
    corpus = TokenCorpus(FILE_TOKEN_CACHE, val_fraction, block_size)

    actual_steps_per_epoch = steps_per_epoch
    if actual_steps_per_epoch is None:
        actual_steps_per_epoch = max(1, len(corpus.train_data) // (batch_size * block_size))
    total_train_steps = max_epochs * actual_steps_per_epoch

    print(f"-> Train tokens: {len(corpus.train_data):,}")
    print(f"-> Val tokens:   {len(corpus.val_data):,}")
    print(f"-> Steps/epoch:  {actual_steps_per_epoch:,}")
    print(f"-> Total steps:  {total_train_steps:,}")

    model = MiniGPT().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    start_global_step, best_val_loss = load_checkpoint_if_exists(model, optimizer, scaler, train_generator)
    if start_global_step >= total_train_steps:
        print("-> Checkpoint đã đạt hoặc vượt total steps hiện tại. Tăng max_epochs nếu muốn train tiếp.")
        sys.exit(0)

    print(f"-> Tham số model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} triệu")
    print("\n[ BẮT ĐẦU HUẤN LUYỆN ]")

    model.train()
    running_loss = 0.0
    running_count = 0
    last_log_time = time.time()
    next_global_step = start_global_step

    try:
        for global_step in range(start_global_step, total_train_steps):
            next_global_step = global_step
            lr = get_lr(global_step, total_train_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = corpus.get_batch("train", batch_size, block_size, train_generator, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            next_global_step = global_step + 1
            running_loss += loss.item()
            running_count += 1

            if (global_step + 1) % log_interval == 0 or global_step == start_global_step:
                elapsed = time.time() - last_log_time
                avg_loss = running_loss / max(1, running_count)
                tokens_per_second = running_count * batch_size * block_size / max(elapsed, 1e-9)
                epoch = global_step // actual_steps_per_epoch + 1
                step_in_epoch = global_step % actual_steps_per_epoch + 1
                print(
                    f"   Epoch {epoch}/{max_epochs} | "
                    f"Step {step_in_epoch:,}/{actual_steps_per_epoch:,} | "
                    f"Global {global_step + 1:,}/{total_train_steps:,} | "
                    f"Loss {avg_loss:.4f} | LR {lr:.2e} | {tokens_per_second:,.0f} tok/s"
                )
                running_loss = 0.0
                running_count = 0
                last_log_time = time.time()

            if (global_step + 1) % eval_interval == 0 or (global_step + 1) == total_train_steps:
                losses = estimate_loss(model, corpus, use_amp, global_step + 1)
                print(f"\n[EVAL] Step {global_step + 1:,} | Train loss {losses['train']:.4f} | Val loss {losses['val']:.4f}")
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(model, optimizer, scaler, global_step + 1, best_val_loss, FILE_BEST_MODEL, train_generator)
                    print(f"[BEST] Val loss mới tốt nhất: {best_val_loss:.4f}")

            if (global_step + 1) % preview_interval == 0:
                print_preview(model, tokenizer)

            if (global_step + 1) % save_interval == 0:
                save_checkpoint(model, optimizer, scaler, global_step + 1, best_val_loss, FILE_SAVE_MODEL, train_generator)

        save_checkpoint(model, optimizer, scaler, total_train_steps, best_val_loss, FILE_SAVE_MODEL, train_generator)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print(f" DỪNG TRAIN. Sẽ resume tại global step {next_global_step:,}")
        print("=" * 50)
        save_checkpoint(model, optimizer, scaler, next_global_step, best_val_loss, FILE_SAVE_MODEL, train_generator)
        print("\n[!] ĐÃ LƯU AN TOÀN. Lần tới code sẽ chạy tiếp đúng vị trí.")
        sys.exit(0)

    print("=" * 50)
    print(" HOÀN TẤT TRAIN BASE MODEL!")
    print("=" * 50)
