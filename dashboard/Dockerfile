FROM nginx:alpine

RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 3650 \
        -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/server.key \
        -out    /etc/nginx/ssl/server.crt \
        -subj   "/C=ES/ST=Local/L=Local/O=monitor-conversacion/CN=localhost"

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80 443 8516

CMD ["nginx", "-g", "daemon off;"]
