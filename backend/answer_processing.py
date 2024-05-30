from start_model_processing import processing_questions
from config import conn
    
def get_answer(question):
    # после получения ответа нужно найти его в БД, и вернуть ответ в исходном виде(не прошедший лемматизацию, обработку стоп слов и токенизацию)
    answer,best_similarity  = processing_questions(question)
    print("\nОтвет, после косинусного сходства: ", answer)
    # Создание объекта курсора для выполнения SQL-запросов
    cur = conn.cursor()

    # Выполнение SQL-запроса с использованием параметра
    cur.execute("SELECT a.id_answer, a.answer FROM answers a WHERE a.answer_processed = %s;", (answer,))
    row = cur.fetchone()
    id_answer, original_answer = row
    print("ID Ответа:", id_answer)
    # Вывод результата
    
    return original_answer, best_similarity