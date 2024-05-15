import requests

# Функция для отправки вопроса в AI сервис и получения ответа
def get_answer_from_ai(question):
    # URL API AI сервиса
    ai_service_url = 'URL AI сервиса'
    
    # Параметры запроса
    params = {
        'question': question,
        # Другие необходимые параметры
    }
    
    # Отправка запроса
    response = requests.post(ai_service_url, json=params)
    
    # Проверка успешности запроса
    if response.status_code == 200:
        # Обработка и возврат ответа
        return response.json()['answer']
    else:
        # Обработка ошибки
        return 'Произошла ошибка при получении ответа от AI сервиса.'

# Пример использования функции
question = 'Какая сегодня погода?'
answer = get_answer_from_ai(question)
print(answer)