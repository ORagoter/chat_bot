import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, trg):
        # src: исходная последовательность
        # trg: целевая последовательность
        # Пропускаем исходную последовательность через энкодер
        _, hidden = self.encoder(src)
        
        # Начальное скрытое состояние декодера
        decoder_input = trg[0, :]
        decoder_input = decoder_input.unsqueeze(0)
        
        for t in range(1, trg.shape[0]):
            output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc_out(output.squeeze(0))
            decoder_input = trg[t, :].unsqueeze(0)
        
        return prediction
