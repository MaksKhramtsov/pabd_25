FROM python:3.11-slim

WORKDIR /

COPY service service
COPY models models
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /service

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]