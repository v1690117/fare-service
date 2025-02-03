FROM python:3.11.4

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/* \

WORKDIR /app

RUN pip install face_recognition
RUN pip install uvicorn gunicorn flask asgiref numpy

COPY server.py server.py
COPY model.pkl model.pkl

EXPOSE 8080

CMD uvicorn 'server:asgi_app' --host=0.0.0.0 --port=8080
