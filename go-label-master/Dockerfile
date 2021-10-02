FROM golang:1.15.0-buster

USER root

WORKDIR /workspace

ENV GOPROXY="https://goproxy.cn" GO111MODULE=on

COPY . .

RUN go mod download && \
    mv /etc/apt/sources.list /etc/apt/sources.list_backup && \
    cp ./sources.list /etc/apt/ && \
    apt-get update && apt-get install -yq --no-install-recommends \
    sqlite3 \
    nginx && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default && \
    rm -f /etc/nginx/nginx.conf && \
    cp /workspace/label.conf /etc/nginx/sites-available/ && \
    cp /workspace/nginx.conf /etc/nginx/ && \
    mkdir -p /workspace/log && \
    cp /workspace/js/* /workspace/www/static/js/

RUN go build -o /workspace/label .
    
EXPOSE 80

ENTRYPOINT ["/workspace/run.sh"] 
