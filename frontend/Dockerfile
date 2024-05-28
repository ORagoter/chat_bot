# bot/frontend/Dockerfile
FROM nginx:alpine

# Удаляем стандартные файлы конфигурации NGINX и добавляем свои
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d

# Копируем статические файлы в директорию сервера
COPY . /usr/share/nginx/html

# Открываем порт
EXPOSE 80