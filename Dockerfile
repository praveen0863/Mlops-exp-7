FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .
COPY app.py .

RUN python train_model.py

EXPOSE 5000

CMD ["python", "app.py"]
