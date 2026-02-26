FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
