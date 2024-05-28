import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from config import conn

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

# Определение класса Seq2Seq с использованием внимания и dropout
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, dropout_p=0.5):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_p)
        
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
            output = self.dropout(output)
            outputs.append(output.squeeze(1))
        outputs = torch.stack(outputs, dim=1)
        return outputs

# Загрузка сохраненной модели и оптимизатора
file_path = 'test_model_third.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

input_dim = len(data['word_index'])
output_dim = len(data['word_index'])
embedding_dim = data['embedding_dim']
hidden_dim = data['hidden_dim']

model = Seq2Seq(input_dim, output_dim, embedding_dim, hidden_dim)
model.load_state_dict(data['model_state_dict'])

optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(data['optimizer_state_dict'])

word_index = data['word_index']
index_word = {v: k for k, v in word_index.items()}

# Функция для преобразования текста в последовательность индексов
def text_to_sequence(text, word_index):
    tokens = word_tokenize(text.lower())
    sequence = [word_index.get(token, word_index['<UNK>']) for token in tokens]
    return sequence

# Функция для генерации ответа
def generate_response(model, input_tensor, max_length, word_index, index_word):
    model.eval()
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder_lstm(model.embedding(input_tensor))
        decoder_input = torch.tensor([[word_index['<PAD>']]]).to(input_tensor.device)
        decoded_words = []

        for _ in range(max_length):
            attn_weights = model.attention(hidden[-1], encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            decoder_input_embedded = model.embedding(decoder_input)
            decoder_input_combined = torch.cat((decoder_input_embedded, attn_applied), 2)
            output, (hidden, cell) = model.decoder_lstm(decoder_input_combined, (hidden, cell))
            output = model.fc(torch.cat((output, attn_applied), 2))
            topv, topi = output.topk(1)
            if topi.item() == word_index['<PAD>']:
                break
            else:
                decoded_words.append(index_word[topi.item()])
            decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)
    return ' '.join(decoded_words)

# Подключение к базе данных и извлечение вопросов и ответов
cur = conn.cursor()
cur.execute("SELECT question, answer FROM processed_data")
rows = cur.fetchall()
cur.close()
conn.close()

# Преобразование результатов в список вопросов и ответов
questions_answers = [(row[0], row[1]) for row in rows]

# Функция для вычисления косинусного сходства между текстами
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

# Генерация ответов и вычисление косинусного сходства
similarities = []
correct_count = 0
for question, true_answer in questions_answers:
    input_sequence = text_to_sequence(question, word_index)
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)
    max_length = 50
    generated_answer = generate_response(model, input_tensor, max_length, word_index, index_word)
    
    # Сравнение с ответами из базы данных
    best_similarity = 0
    best_answer = None
    for _, answer in questions_answers:
        similarity = calculate_similarity(generated_answer, answer)
        if similarity > best_similarity:
            best_similarity = similarity
            best_answer = answer
    
    is_correct = best_answer.strip() == true_answer.strip()
    similarities.append((best_similarity, is_correct))
    if is_correct:
        correct_count += 1
    print(f"Вопрос: {question}")
    print(f"Сгенерированный ответ: {generated_answer}")
    print(f"Самый подходящий ответ: {best_answer}")
    print(f"Правильный ответ: {true_answer}")
    print(f"Косинусное сходство: {best_similarity}")
    print(f"Правильный: {is_correct}")
    print("=======")

# Разделение сходств на правильные и неправильные
correct_similarities = [sim for sim, is_correct in similarities if is_correct]
incorrect_similarities = [sim for sim, is_correct in similarities if not is_correct]

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.hist(correct_similarities, bins=20, alpha=0.7, label='Правильные ответы', color='green')
plt.hist(incorrect_similarities, bins=20, alpha=0.7, label='Неправильные ответы', color='red')
plt.xlabel('Косинусное сходство')
plt.ylabel('Частота')
plt.title('Распределение косинусного сходства между сгенерированными и правильными ответами')
plt.legend()
plt.show()

# Вывод количества правильных ответов
print(f"Количество правильных ответов: {correct_count} из {len(questions_answers)}")
