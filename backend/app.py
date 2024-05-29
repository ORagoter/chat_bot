from flask import Flask, request, jsonify
from flask_cors import CORS

from answer_processing import get_answer

app = Flask(__name__)
CORS(app, resources={r"/bot": {"origins": "*"}})
     
@app.route('/bot', methods=['POST'])
def handle_data():
    # Получаем данные из запроса в формате JSON
    request_data = request.json
    # Извлекаем переменную data из полученных данных
    input_text = request_data.get('data')
    # Обработка полученных данных
    print(f"Полученный вопрос: {input_text}")
    answer = get_answer(input_text.strip())
    # Возвращаем ответ
    print(f"Отправляемый ответ: {answer}")
    return jsonify({"answer": answer})

if __name__ == '__main__':
    pass
