from flask import Flask, request, jsonify
from flask_cors import CORS

from processing_question import processing_questions

app = Flask(__name__)
CORS(app)

@app.route('/bot', methods=['POST'])
def handle_data():
      # Получаем данные из запроса в формате JSON
    request_data = request.json
    # Извлекаем переменную data из полученных данных
    input_text = request_data.get('data')
    # Обработка полученных данных
    print(input_text)
    answer = processing_questions(input_text)
    # Возвращаем ответ
    print(answer)
    return jsonify({"answer":answer})
    

if __name__ == '__main__':
    app.run(debug=True)
    