import psycopg2
def processing_questions(quest):
    # Параметры подключения к базе данных
    conn = psycopg2.connect(
        dbname='postgres', #имя базы данных
        user='postgres.zzyahwklsrihlglqbsfd', # имя пользователя
        password='xy9$G/Cy~b~)&+_', # пароль
        host='aws-0-eu-central-1.pooler.supabase.com',  # хост
        port='5432' # порт
    )

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
        print('Ответ не найден.')

    # Закрытие соединения с базой данных
    cur.close()
    conn.close()
    
    return answer