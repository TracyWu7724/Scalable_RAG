FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY server/ ./server/
COPY server/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
