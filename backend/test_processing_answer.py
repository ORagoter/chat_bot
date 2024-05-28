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