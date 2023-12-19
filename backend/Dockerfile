# Choose the appropriate PyTorch image. This example uses the CPU version.
# If you need CUDA support, select a version that includes CUDA.
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Copy your application's files into the container
COPY . .

# Install any additional dependencies not included in the base image
RUN pip install flask
RUN pip install flask-cors

# Expose the port your application runs on
EXPOSE 5000

# Run your application
CMD ["python", "main.py"]