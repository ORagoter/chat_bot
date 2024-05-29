# bot/backend/Dockerfile
FROM python:3.12.2



# Создаем рабочую директорию
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код в контейнер
COPY . .

# Указываем переменную среды для Flask
ENV FLASK_APP=app.py

# Запускаем приложение с помощью Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
