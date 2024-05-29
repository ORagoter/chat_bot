# bot/backend/Dockerfile
FROM python:3.12.2-slim

# Установим рабочую директорию в контейнере
WORKDIR /app

# Скопируем все файлы из текущей директории в контейнер
COPY . .

# Установка переменной среды PATH для поиска pg_config
ENV PATH="${PATH}:D:\PostgreSQL\bin"
# Установим зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["python", "app.py"]
