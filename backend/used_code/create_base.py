import psycopg2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

import config
# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Загрузка русского списка стоп-слов
stop_words = set(stopwords.words('russian'))

# Инициализация морфологического анализатора pymorphy3
morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    text = text.lower()  # Приводим текст к нижнему регистру
    tokens = word_tokenize(text)  # Токенизация текста
    tokens = [word for word in tokens if word not in stop_words]  # Удаляем стоп-слова
    tokens = [word for word in tokens if word.isalnum()]  # Удаляем небуквенно-цифровые символы
    tokens = [morph.parse(word)[0].normal_form for word in tokens]  # Лемматизация
    preprocessed_text = ' '.join(tokens)  # Объединяем токены обратно в строку
    return preprocessed_text

# Устанавливаем соединение с базой данных Supabase
conn = config.conn

# Создаем объект cursor для выполнения SQL-запросов
cur = conn.cursor()

# Выполняем SQL-запрос для извлечения данных из обеих таблиц
cur.execute("""
SELECT b.name_of_block, q.question, q.answer
FROM blocks b
JOIN question q ON b.number_of_block = q.number_of_block
""")

# Получаем все строки результата запроса
rows = cur.fetchall()

# Преобразуем результат запроса в удобный формат, например, список словарей
data = [{"name_of_block": row[0], "question": row[1], "answer": row[2]} for row in rows]

# Применяем предварительную обработку к каждой записи
for item in data:
    item["name_of_block"] = preprocess_text(item["name_of_block"])  # Обработка названия блока
    item["question"] = preprocess_text(item["question"])
    item["answer"] = preprocess_text(item["answer"])

# Создаем новую таблицу в Supabase для хранения обработанных данных с уникальным идентификатором
cur.execute("""
CREATE TABLE IF NOT EXISTS processed_data (
    id SERIAL PRIMARY KEY,
    name_of_block TEXT,
    question TEXT,
    answer TEXT
)
""")

# Вставляем предварительно обработанные данные в новую таблицу
for item in data:
    cur.execute("INSERT INTO processed_data (name_of_block, question, answer) VALUES (%s, %s, %s)",
                (item["name_of_block"], item["question"], item["answer"]))

# Фиксируем изменения и закрываем соединение
conn.commit()
cur.close()
conn.close()
