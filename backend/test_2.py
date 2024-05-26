import nltk
import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils
from nltk.tokenize import word_tokenize
from collections import defaultdict
import os
from config import conn  # Подключение к базе данных

# Функция предварительной обработки текста
def preprocess_text(text):
    # Здесь можно добавить удаление пунктуации, приведение к нижнему регистру и другие преобразования
    return text

# Функция для построения словаря
def build_vocab(texts):
    word_index = defaultdict(lambda: len(word_index))
    word_index['<PAD>'] = 0  # Индекс для заполнения (padding)
    word_index['<UNK>'] = 1  # Индекс для неизвестных слов (unknown)

    for text in texts:
        tokens = word_tokenize(text.lower())  # Токенизация текста
        for token in tokens:
            if token not in word_index:
                word_index[token] = len(word_index)  # Добавляем новые токены в словарь
    return word_index

# Функция для преобразования текста в числовые последовательности
def text_to_sequence(text, word_index):
    tokens = word_tokenize(text.lower())  # Токенизация текста
    sequence = [word_index[token] if token in word_index else word_index['<UNK>'] for token in tokens]
    return sequence

# Подключение к базе данных и загрузка данных
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

# Построение словаря на основе всех текстов (вопросы + ответы)
all_texts = preprocessed_train_questions + preprocessed_train_answers
word_index = build_vocab(all_texts)

# Преобразование текстов в числовые последовательности
train_questions_seq = [text_to_sequence(text, word_index) for text in preprocessed_train_questions]
train_answers_seq = [text_to_sequence(text, word_index) for text in preprocessed_train_answers]
test_questions_seq = [text_to_sequence(text, word_index) for text in preprocessed_test_questions]
test_answers_seq = [text_to_sequence(text, word_index) for text in preprocessed_test_answers]

# Определение класса Attention для механизма внимания
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)  # Линейный слой для внимания
        self.v = nn.Parameter(torch.rand(hidden_dim))  # Параметр внимания

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)  # Повторяем скрытое состояние для каждого таймстепа
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), 2)))  # Вычисляем энергию внимания
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_weights = torch.bmm(v, energy).squeeze(1)  # Вычисляем веса внимания
        return torch.softmax(attention_weights, dim=1)  # Применяем softmax для получения вероятностей

# Определение класса Seq2Seq с использованием внимания
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # Слой для эмбеддингов
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM слой энкодера
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)  # LSTM слой декодера
        self.attention = Attention(hidden_dim)  # Механизм внимания
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Линейный слой для предсказаний

    def forward(self, src, trg):
        embedded_src = self.embedding(src)  # Эмбеддинги для входной последовательности
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)  # Выходы и состояния энкодера

        outputs = []
        for t in range(trg.size(1)):
            embedded_trg = self.embedding(trg[:, t]).unsqueeze(1)  # Эмбеддинги для текущего шага декодера
            attn_weights = self.attention(hidden[-1], encoder_outputs)  # Вычисляем веса внимания
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # Применяем внимание к выходам энкодера

            decoder_input = torch.cat((embedded_trg, attn_applied), 2)  # Объединяем эмбеддинги и внимание
            output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))  # Выход декодера

            output = self.fc(torch.cat((output, attn_applied), 2))  # Предсказания
            outputs.append(output.squeeze(1))

        outputs = torch.stack(outputs, dim=1)
        return outputs

# Функция для создания батчей данных
def create_batch(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Гиперпараметры
input_dim = len(word_index)
output_dim = len(word_index)
embedding_dim = 100
hidden_dim = 128
batch_size = 32

# Инициализация модели, функции потерь и оптимизатора
model = Seq2Seq(input_dim, output_dim, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss(ignore_index=word_index['<PAD>'])
optimizer = torch.optim.Adam(model.parameters())

# Преобразование данных в тензоры
train_questions_tensor = [torch.tensor(seq) for seq in train_questions_seq]
train_answers_tensor = [torch.tensor(seq) for seq in train_answers_seq]
test_questions_tensor = [torch.tensor(seq) for seq in test_questions_seq]
test_answers_tensor = [torch.tensor(seq) for seq in test_answers_seq]

# Цикл обучения модели
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Перевод модели в режим обучения
    epoch_loss = 0
    
    # Проход по обучающим данным батчами
    for q_batch, a_batch in zip(create_batch(train_questions_tensor, batch_size), create_batch(train_answers_tensor, batch_size)):
        q_batch = rnn_utils.pad_sequence(q_batch, batch_first=True, padding_value=word_index['<PAD>'])  # Заполнение батча
        a_batch = rnn_utils.pad_sequence(a_batch, batch_first=True, padding_value=word_index['<PAD>'])  # Заполнение батча
        
        optimizer.zero_grad()  # Обнуление градиентов
        output = model(q_batch, a_batch[:, :-1])  # Прогон модели
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # Преобразование выхода
        a_batch = a_batch[:, 1:].contiguous().view(-1)  # Преобразование таргета
        
        loss = criterion(output, a_batch)  # Вычисление потерь
        loss.backward()  # Обратное распространение
        optimizer.step()  # Обновление параметров
        
        epoch_loss += loss.item()
    
    print(f"Эпоха {epoch+1}/{num_epochs}, Потери: {epoch_loss/len(train_questions_tensor)}")

# Функция для оценки модели на тестовых данных
def evaluate(model, test_questions, test_answers):
    model.eval()  # Перевод модели в режим оценки
    epoch_loss = 0
    with torch.no_grad():  # Отключение вычисления градиентов
        for q_batch, a_batch in zip(create_batch(test_questions, batch_size), create_batch(test_answers, batch_size)):
            q_batch = rnn_utils.pad_sequence(q_batch, batch_first=True, padding_value=word_index['<PAD>'])  # Заполнение батча
            a_batch = rnn_utils.pad_sequence(a_batch, batch_first=True, padding_value=word_index['<PAD>'])  # Заполнение батча

            output = model(q_batch, a_batch[:, :-1])  # Прогон модели
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # Преобразование выхода
            a_batch = a_batch[:, 1:].contiguous().view(-1)  # Преобразование таргета
            
            loss = criterion(output, a_batch)  # Вычисление потерь
            epoch_loss += loss.item()
    
    return epoch_loss / len(test_questions_tensor)

# Оценка модели на тестовых данных
test_loss = evaluate(model, test_questions_tensor, test_answers_tensor)
print(f"Потери на тестовых данных: {test_loss}")

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Пример обученной модели и оптимизатора
model = model()  # Ваша обученная модель Seq2Seq
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Пример оптимизатора

# Пример данных, которые нужно сохранить
data = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'word_index': word_index
}

# Путь к файлу для сохранения данных
file_path = 'saved_model.pkl'

# Сохранение данных в файл
with open(file_path, 'wb') as file:
    pickle.dump(data, file)

print("Модель успешно сохранена в файл:", file_path)