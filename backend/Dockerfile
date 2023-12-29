# Choose the appropriate PyTorch image. This example uses the CPU version.
# If you need CUDA support, select a version that includes CUDA.
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Copy your application's files into the container
COPY . .

RUN pip install flask
RUN pip install flask-cors
RUN pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
ENV HOST 0.0.0.0
ENV PORT 8080
EXPOSE 8080

# Run your application
CMD ["python", "main.py"]