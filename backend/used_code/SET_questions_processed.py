import nltk
from nltk.corpus import stopwords
import pymorphy3
import re

# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Загрузка русского списка стоп-слов
stop_words = set(stopwords.words('russian'))

# Инициализация морфологического анализатора pymorphy3
morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    # Токенизация текста
    tokens = nltk.word_tokenize(text.lower())
    
    # Лемматизация и удаление стоп-слов
    lemmatized_tokens = []
    for token in tokens:
        # Исключаем пунктуацию и ненужные символы
        token = re.sub(r'[^а-яА-Яё]', '', token)
        # Если токен не является пустым и не является стоп-словом
        if token and token not in stop_words:
            # Лемматизация
            lemmatized_token = morph.parse(token)[0].normal_form
            lemmatized_tokens.append(lemmatized_token)
    
    # Сборка предложения из лемматизированных токенов
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text




import psycopg2

# Устанавливаем параметры подключения
conn_params = {
    'dbname': 'postgres',
    'user': 'postgres.zzyahwklsrihlglqbsfd',
    'password': 'xy9$G/Cy~b~)&+_',
    'host': 'aws-0-eu-central-1.pooler.supabase.com',
    'port': '5432'
}

# Устанавливаем соединение
conn = psycopg2.connect(**conn_params)

# Создаем курсор для выполнения запросов
cur = conn.cursor()

# Выполняем SQL-запрос для извлечения данных
select_query = "SELECT id_question, question FROM questions"
cur.execute(select_query)

# Получаем все строки результата
rows = cur.fetchall()

# Обрабатываем данные в Python
processed_rows = []
for row in rows:
    id_question, question = row
    processed_question = preprocess_text(question)
    processed_rows.append((processed_question, id_question))

# Выполняем SQL-запрос для загрузки обработанных данных обратно в базу данных
update_query = "UPDATE questions SET question_processed = %s WHERE id_question = %s"
cur.executemany(update_query, processed_rows)

# Подтверждаем изменения
conn.commit()

# Закрываем курсор и соединение
cur.close()
conn.close()
