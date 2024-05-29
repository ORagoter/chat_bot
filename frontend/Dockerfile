# Use an official Nginx image
FROM nginx:alpine

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the frontend code into the container
COPY . /usr/share/nginx/html