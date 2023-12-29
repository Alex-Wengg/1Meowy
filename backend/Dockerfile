FROM python:3.9-slim

COPY . .

RUN pip install --no-cache-dir flask flask-cors \
    && pip install torch==2.0.0 --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

ENV HOST 0.0.0.0
ENV PORT 8080
EXPOSE 8080

CMD ["python", "main.py"]
