import torch
import torch.nn as nn

VOCAB = 5000
EMBED = 128
HIDDEN = 256
SEQ_LEN = 40


class TextClassifier(nn.Module):
    """Bidirectional LSTM sentiment classifier: embed tokens, read them in both directions, classify.
    zh: 双向 LSTM 情感分类器：词嵌入后双向读取序列，再分类。"""

    def __init__(self, vocab=VOCAB, embed=EMBED, hidden=HIDDEN, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, tokens):
        x = self.embedding(tokens)          # [B, L, E]
        out, (h, c) = self.lstm(x)          # out: [B, L, 2H]
        x = out.mean(dim=1)                 # average over time
        return self.fc(self.dropout(x))


if __name__ == "__main__":
    model = TextClassifier()
    tokens = torch.randint(0, VOCAB, (1, SEQ_LEN))
    logits = model(tokens)
