import torch
import torch.optim as optim
import pickle
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import conn
from model_seq import Seq2Seq
from test_processing_answer import preprocess_text
# В данном файле происходит обработка вопроса с помощью сохранённой модели, функция processing_questions возвращает ответ


# обработка вопроса
def processing_questions(input_text):
    print("Полученный вопрос: ", input_text)
    preprocessed_text = preprocess_text(input_text)
    print("Обработанный вопрос:", preprocessed_text)
    input_sequence = text_to_sequence(preprocessed_text, word_index)
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Добавляем размер батча

    # Генерация ответа
    print("Генерация ответа.")
    max_length = 50
    response = generate_response(model, input_tensor, max_length, word_index, index_word)
    print("Ответ успешно сгенерирован: \n", response, end= "\n\n")
    
    # Подключение к базе данных и извлечение ответов
    cur = conn.cursor()
    cur.execute("SELECT answer FROM processed_data")
    rows = cur.fetchall()
    print("Успешное подключение к базе данных")
    
    # Преобразование результатов в список ответов
    database_answers = [row[0] for row in rows]

    
    print('Оценка сгенерированного ответа по сходству с ответами из базы данных.')
    best_similarity = 0
    best_answer = None
    for answer in database_answers:
        similarity = calculate_similarity(response, answer)
        if similarity > best_similarity:
            best_similarity = similarity
            best_answer = answer
    print("Косинусная точность: ", best_similarity)
    
    return best_answer




file_path = 'backend/models/main_model.pkl'
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
        decoder_input = torch.tensor([[word_index['<PAD>']]]).to(input_tensor.device)  # Начало последовательности
        decoded_words = []

        for _ in range(max_length):
            # Вычисление весов внимания
            attn_weights = model.attention(hidden[-1], encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            # Встраивание текущего входа декодера
            decoder_input_embedded = model.embedding(decoder_input)
            decoder_input_combined = torch.cat((decoder_input_embedded, attn_applied), 2)
            # Декодирование
            output, (hidden, cell) = model.decoder_lstm(decoder_input_combined, (hidden, cell))
            output = model.fc(torch.cat((output, attn_applied), 2))
            # Определение следующего слова
            topv, topi = output.topk(1)
            if topi.item() == word_index['<PAD>']:
                break
            else:
                decoded_words.append(index_word[topi.item()])
            # Обновление входа для следующего шага декодера
            decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)
    return ' '.join(decoded_words)


# Функция для вычисления косинусного сходства между текстами
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]


    

