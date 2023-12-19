FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]
