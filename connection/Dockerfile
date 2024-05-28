# bot/connection/Dockerfile
FROM node:14

# Установим рабочую директорию в контейнере
WORKDIR /app

# Скопируем все файлы из текущей директории в контейнер
COPY . .

# Установим зависимости (предположим, что у вас есть package.json и package-lock.json)
RUN npm install

# Открываем порт
EXPOSE 3000

# Запуск приложения
CMD ["node", "app.js"]
