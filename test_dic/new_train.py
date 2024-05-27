import nltk
import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils
from nltk.tokenize import word_tokenize
import pickle
import torch.optim as optim
from config import conn  # Подключение к базе данных

# Функция предварительной обработки текста
def preprocess_text(text):
    return text

# Функция для построения словаря
def build_vocab(texts):
    word_index = {'<PAD>': 0, '<UNK>': 1}
    current_index = len(word_index)
    
    for text in texts:
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token not in word_index:
                word_index[token] = current_index
                current_index += 1
                
    return word_index

# Функция для преобразования текста в числовые последовательности
def text_to_sequence(text, word_index):
    tokens = word_tokenize(text.lower())
    sequence = [word_index.get(token, word_index['<UNK>']) for token in tokens]
    return sequence

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

            decoder_input = torch.cat((embedded_trg, attn_applied), 2)
            output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))

            output = self.fc(torch.cat((output, attn_applied), 2))
            outputs.append(output.squeeze(1))

        outputs = torch.stack(outputs, dim=1)
        return outputs

# Функция для создания батчей данных
def create_batch(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Загрузка модели из файла
file_path = 'saved_model.pkl'
with open(file_path, 'rb') as file:
    model_data = pickle.load(file)

# Извлечение параметров модели
input_dim = model_data['input_dim']
output_dim = model_data['output_dim']
embedding_dim = model_data['embedding_dim']
hidden_dim = model_data['hidden_dim']
word_index = model_data['word_index']

# Создание и инициализация модели
model = Seq2Seq(input_dim, output_dim, embedding_dim, hidden_dim)
model.load_state_dict(model_data['model_state_dict'])

# Настройка оптимизатора и функции потерь
criterion = nn.CrossEntropyLoss(ignore_index=word_index['<PAD>'])
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(model_data['optimizer_state_dict'])

# Загрузка данных
cur = conn.cursor()
cur.execute("SELECT name_of_block, question, answer FROM processed_data")
rows = cur.fetchall()
cur.close()

# Разделение данных на обучающую и тестовую выборки (70/30)
train_size = int(0.7 * len(rows))
train_data = rows[:train_size]
test_data = rows[train_size:]

# Извлечение вопросов и ответов из обучающей и тестовой выборок
train_questions = [row[1] for row in train_data]
train_answers = [row[2] for row in train_data]
test_questions = [row[1] for row in test_data]
test_answers = [row[2] for row in test_data]

# Предварительная обработка текстов вопросов и ответов
preprocessed_train_questions = [preprocess_text(question) for question in train_questions]
preprocessed_train_answers = [preprocess_text(answer) for answer in train_answers]
preprocessed_test_questions = [preprocess_text(question) for question in test_questions]
preprocessed_test_answers = [preprocess_text(answer) for answer in test_answers]

# Преобразование текстов в числовые последовательности
train_questions_seq = [text_to_sequence(text, word_index) for text in preprocessed_train_questions]
train_answers_seq = [text_to_sequence(text, word_index) for text in preprocessed_train_answers]
test_questions_seq = [text_to_sequence(text, word_index) for text in preprocessed_test_questions]
test_answers_seq = [text_to_sequence(text, word_index) for text in preprocessed_test_answers]

# Преобразование данных в тензоры
train_questions_tensor = [torch.tensor(seq) for seq in train_questions_seq]
train_answers_tensor = [torch.tensor(seq) for seq in train_answers_seq]
test_questions_tensor = [torch.tensor(seq) for seq in test_questions_seq]
test_answers_tensor = [torch.tensor(seq) for seq in test_answers_seq]

# Цикл дообучения модели
num_epochs = 20
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for q_batch, a_batch in zip(create_batch(train_questions_tensor, batch_size), create_batch(train_answers_tensor, batch_size)):
        q_batch = rnn_utils.pad_sequence(q_batch, batch_first=True, padding_value=word_index['<PAD>'])
        a_batch = rnn_utils.pad_sequence(a_batch, batch_first=True, padding_value=word_index['<PAD>'])
        
        optimizer.zero_grad()
        output = model(q_batch, a_batch[:, :-1])
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        a_batch = a_batch[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, a_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Эпоха {epoch+1}/{num_epochs}, Потери: {epoch_loss/len(train_questions_tensor)}")

# Функция для оценки модели на тестовых данных
def evaluate(model, test_questions, test_answers):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for q_batch, a_batch in zip(create_batch(test_questions_tensor, batch_size), create_batch(test_answers_tensor, batch_size)):
            q_batch = rnn_utils.pad_sequence(q_batch, batch_first=True, padding_value=word_index['<PAD>'])
            a_batch = rnn_utils.pad_sequence(a_batch, batch_first=True, padding_value=word_index['<PAD>'])

            output = model(q_batch, a_batch[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            a_batch = a_batch[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, a_batch)
            epoch_loss += loss.item()
    
    return epoch_loss / len(test_questions_tensor)

# Оценка модели на тестовых данных
test_loss = evaluate(model, test_questions_tensor, test_answers_tensor)
print(f"Потери на тестовых данных: {test_loss}")

# Сохранение обновленной модели и оптимизатора
model_data = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'word_index': word_index,
    'input_dim': input_dim,
    'output_dim': output_dim,
    'embedding_dim': embedding_dim,
    'hidden_dim': hidden_dim
}

with open(file_path, 'wb') as file:
    pickle.dump(model_data, file)

print("Модель успешно сохранена в файл:", file_path)
