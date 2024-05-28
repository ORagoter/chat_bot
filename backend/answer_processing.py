from backend.start_model_processing import processing_questions
from config import conn
    
def get_answer(question):
    # после получения ответа нужно найти его в БД, и вернуть ответ в исходном виде(не прошедший лемматизацию, обработку стоп слов и токенизацию)
    answer = processing_questions(question)
    print("\nОтвет, после косинусного сходства: ", answer)
    # Создание объекта курсора для выполнения SQL-запросов
    cur = conn.cursor()

    # SQL-запрос для получения id вопроса из таблицы processed_data по ответу
    cur.execute("SELECT id FROM processed_data WHERE answer = %s", (answer,))
    processed_data_id = cur.fetchone()[0]


    print("ID Ответа:", processed_data_id)
    # SQL-запрос для получения вопроса из таблицы question по id
    cur.execute("SELECT answer FROM question WHERE id = %s", (processed_data_id,))
    out_answer = cur.fetchone()[0]


    # Вывод результата
    print("Ответ из БД:", out_answer)
    
    return out_answer