# bot/backend/Dockerfile
FROM python:3.8-slim

# Установим рабочую директорию в контейнере
WORKDIR /app

# Скопируем все файлы из текущей директории в контейнер
COPY . .

# Установим зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["python", "app.py"]
