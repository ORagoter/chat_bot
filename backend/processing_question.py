import psycopg2
import config
def processing_questions(quest):
    # Параметры подключения к базе данных
    conn = config.conn

    # Создаем курсор для выполнения операций с базой данных
    cur = conn.cursor()

    # SQL-запрос для получения ответа на вопрос
    query = """
    SELECT answer
    FROM question
    WHERE question = %s;
    """

    # Вопрос, на который нужно получить ответ
    question_to_find = quest

    # Выполнение запроса
    cur.execute(query, (question_to_find,))

    # Получение результата
    answer = cur.fetchone()

    if answer:
        print(f'Ответ на вопрос: {answer[0]}')
    else:
        answer = "Ответ не найден"

    # Закрытие соединения с базой данных
    cur.close()
    conn.close()
    
    return answer