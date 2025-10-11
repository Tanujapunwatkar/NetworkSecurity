FROM python:3.12-slim-bookworm

WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install -y awscli

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
