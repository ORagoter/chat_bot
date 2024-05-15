from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/bot', methods=['POST'])
def handle_data():
    data = request.json
    # Обработка полученных данных
    processed_data = data['data']  # Пример обработки
    # Возвращаем ответ
    return jsonify({'result': processed_data})

if __name__ == '__main__':
    app.run(debug=True)
    