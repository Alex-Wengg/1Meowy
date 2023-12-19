FROM python:3.9

COPY . .

RUN pip install flask
RUN pip install flask-cors
RUN pip install torch torchvision torchaudio

EXPOSE 5000

CMD ["python", "main.py"]
