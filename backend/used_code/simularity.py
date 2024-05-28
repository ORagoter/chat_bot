from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    # Создание TF-IDF векторизатора
    vectorizer = TfidfVectorizer()

    # Преобразование текстов в TF-IDF векторы
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Вычисление косинусного сходства между векторами
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return similarity[0][0]

# Пример использования функции
text1 = "Привет мир"
text2 = "Привет мир да"
similarity_score = calculate_similarity(text1, text2)
print("Косинусное сходство:", similarity_score)