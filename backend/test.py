from config import conn
import psycopg2
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict



# Создаем объект cursor для выполнения SQL-запросов
cur = conn.cursor()

# Выполняем SQL-запрос для извлечения данных из таблицы processed_data
cur.execute("SELECT name_of_block, question, answer FROM processed_data")

# Получаем все строки результата запроса
rows = cur.fetchall()

# Закрываем соединение и курсор
cur.close()
conn.close()

    
    
# Разделение данных на обучающий и тестовый наборы (70/30)
train_data, test_data = train_test_split(rows, test_size=0.3, random_state=42)


print("Размер обучающего набора данных:", len(train_data))
print("Размер тестового набора данных:", len(test_data))







# Загрузка ресурсов NLTK
nltk.download('punkt')

# Пример предварительной обработки текста
def preprocess_text(text):
    tokens = word_tokenize(text)  # Токенизируем текст
    # Дополнительные операции предобработки
    return tokens

# Построение словаря
def build_vocab(texts):
    word_index = defaultdict(lambda: len(word_index))  # Создаем словарь с автоинкрементом индексов
    word_index['<PAD>'] = 0  # Добавляем индекс для заполнения (padding)
    word_index['<UNK>'] = 1  # Добавляем индекс для неизвестных слов (unknown)
    vocab_size = 2  # Начальный размер словаря (заполнитель и неизвестные слова)

    # Проход по всем текстам для построения словаря
    for text in texts:
        tokens = word_tokenize(text.lower())  # Токенизация текста
        for token in tokens:
            if token not in word_index:
                word_index[token] = len(word_index)  # Добавляем новые токены в словарь
    return word_index


# Предобработка и построение словаря для всех вопросов и ответов
questions = [row[1] for row in rows]
answers = [row[2] for row in rows]
preprocessed_questions = [question for question in questions]
preprocessed_answers = [answer for answer in answers]

all_texts = preprocessed_questions + preprocessed_answers
word_index = build_vocab(all_texts)

# Преобразование в числовые последовательности
def text_to_sequence(text, word_index):
    sequence = [word_index[token] if token in word_index else word_index['<UNK>'] for token in text]  # Преобразование в числовую последовательность
    return sequence

# Пример преобразования текста в числовую последовательность
question_sequence = text_to_sequence(preprocessed_questions[0], word_index)
answer_sequence = text_to_sequence(preprocessed_answers[0], word_index)

print("Числовая последовательность вопроса:", question_sequence)
print("Числовая последовательность ответа:", answer_sequence)




# Преобразование всех вопросов и ответов в числовые последовательности
question_sequences = [text_to_sequence(preprocess_text(question), word_index) for question in questions]
answer_sequences = [text_to_sequence(preprocess_text(answer), word_index) for answer in answers]

# Разделение на обучающий и тестовый наборы данных (70/30)
train_questions, test_questions, train_answers, test_answers = train_test_split(
    question_sequences, answer_sequences, test_size=0.3, random_state=42
)

print(f"Размер обучающего набора данных: {len(train_questions)}")
print(f"Размер тестового набора данных: {len(test_questions)}")


print(train_questions)

