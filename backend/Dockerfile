FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY . .

RUN pip install flask
RUN pip install flask-cors

EXPOSE 5000

CMD ["python", "main.py"]
