import torch
import pickle
from nltk.tokenize import word_tokenize
import torch.nn as nn

# Определение класса Attention для механизма внимания
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_weights = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention_weights, dim=1)

# Определение класса Seq2Seq с использованием внимания
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        outputs = []
        for t in range(trg.size(1)):
            embedded_trg = self.embedding(trg[:, t]).unsqueeze(1)
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

            decoder_input = torch.cat((embedded_trg, attn_applied.unsqueeze(1)), 2)  # Добавляем измерение к attn_applied
            output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))

            output = self.fc(torch.cat((output, attn_applied), 2))
            outputs.append(output.squeeze(1))

        outputs = torch.stack(outputs, dim=1)
        return outputs

# Путь к файлу с сохраненной моделью
file_path = 'saved_model.pkl'

# Загрузка модели и оптимизатора
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Инициализация модели
input_dim = len(data['word_index'])
output_dim = len(data['word_index'])
embedding_dim = 100
hidden_dim = 128

model = Seq2Seq(input_dim, output_dim, embedding_dim, hidden_dim)
model.load_state_dict(data['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(data['optimizer_state_dict'])

word_index = data['word_index']
index_word = {v: k for k, v in word_index.items()}

def preprocess_text(text):
    return text

def text_to_sequence(text, word_index):
    tokens = word_tokenize(text.lower())
    sequence = [word_index.get(token, word_index['<UNK>']) for token in tokens]
    return sequence

# Пример входного текста
input_text = "Ваш вопрос здесь"

# Предварительная обработка и преобразование текста
preprocessed_text = preprocess_text(input_text)
input_sequence = text_to_sequence(preprocessed_text, word_index)
input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Добавляем размер батча

def generate_response(model, input_tensor, max_length, word_index, index_word):
    model.eval()
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder_lstm(model.embedding(input_tensor))

        decoder_input = torch.tensor([[word_index['<PAD>']]]).to(input_tensor.device)  # Начало последовательности
        decoded_words = []

        for _ in range(max_length):
            attn_weights = model.attention(hidden[-1], encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

            decoder_input_embedded = model.embedding(decoder_input)
            decoder_input_embedded = decoder_input_embedded.unsqueeze(1)  # Добавляем размер batch_size
            decoder_input_combined = torch.cat((decoder_input_embedded.squeeze(2), attn_applied.squeeze(2)), dim=2)



            output, (hidden, cell) = model.decoder_lstm(decoder_input_combined, (hidden, cell))
            output = model.fc(torch.cat((output, attn_applied), 2)).squeeze(1)

            topv, topi = output.topk(1)
            if topi.item() == word_index['<PAD>']:
                break
            else:
                decoded_words.append(index_word[topi.item()])

            decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)

    return ' '.join(decoded_words)

# Генерация ответа
max_length = 50  # Максимальная длина ответа
response = generate_response(model, input_tensor, max_length, word_index, index_word)
print("Ответ:", response)
