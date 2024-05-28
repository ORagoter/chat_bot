import torch
from torch import nn

# Определение класса Attention для механизма внимания
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # Повторяем hidden state для всех временных шагов
        timestep = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)
        # Вычисляем энергию внимания
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        # Вычисляем вес внимания
        attention_weights = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention_weights, dim=1)

# Определение класса Seq2Seq с использованием внимания и dropout
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, dropout_p = 0.5):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_p) # для создания dropout слоя
        
    def forward(self, src, trg):
        # Встраивание входной последовательности
        embedded_src = self.embedding(src)
        # Кодирование входной последовательности с помощью LSTM
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        outputs = []
        for t in range(trg.size(1)):
            # Встраивание выходной последовательности на текущем временном шаге
            embedded_trg = self.embedding(trg[:, t]).unsqueeze(1)
            # Вычисление весов внимания
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            # Создание входа для декодера
            decoder_input = torch.cat((embedded_trg, attn_applied), 2)
            # Декодирование
            output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            # Полносвязный слой для получения конечного вывода
            output = self.fc(torch.cat((output, attn_applied), 2))
            output = self.dropout(output)  # для применения dropout
            outputs.append(output.squeeze(1))
        # Объединение всех выходов в одну тензорную последовательность
        outputs = torch.stack(outputs, dim=1)
        return outputs