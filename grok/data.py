import torch
from torch.utils.data import Dataset, DataLoader, random_split
import itertools

# 词表定义
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
OP_TOKEN = "+"  # 乘法/加法
MODULUS = 113
# 0-96 的数字转字符串
NUMS = [str(i) for i in range(MODULUS)]
# 词表顺序: <eos>, =, +, 0, 1, ... 96
VOCAB = [EOS_TOKEN, EQ_TOKEN, OP_TOKEN] + NUMS

class ArithmeticDataset(Dataset):
    def __init__(self, modulus=97, seq_len=5):
        self.modulus = modulus
        self.seq_len = seq_len
        self.vocab = VOCAB
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}
        
        self.data = self._generate_data()
        
    def _generate_data(self):
        data = []
        # 生成全排列: a + b = c
        for a in range(self.modulus):
            for b in range(self.modulus):
                c = (a + b) % self.modulus
                # 格式: <eos> a + b = c <eos>
                # 输入: <eos> a + b = c
                # 目标: a + b = c <eos>
                seq = [EOS_TOKEN, str(a), OP_TOKEN, str(b), EQ_TOKEN, str(c), EOS_TOKEN]
                
                # 转换为 ID
                encoded = [self.stoi[s] for s in seq]
                data.append(torch.tensor(encoded, dtype=torch.long))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data[idx]
        # x: 0:-1, y: 1:end
        return full_seq[:-1], full_seq[1:]

def get_dataloaders(args):
    dataset = ArithmeticDataset(modulus=args.modulus)
    
    total_len = len(dataset)
    train_len = int(total_len * args.train_pct)
    val_len = total_len - train_len
    
    # 这里的 generator 保证了基于 seed 的随机切分可复现
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # num_workers=0 避免多进程带来的随机性问题，保证绝对复现
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, dataset.vocab