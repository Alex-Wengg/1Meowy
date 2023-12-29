FROM python:3.9-alpine

# Add glibc compatibility to Alpine
RUN apk --no-cache add ca-certificates wget \
    && wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub \
    && wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.31-r0/glibc-2.31-r0.apk \
    && apk add glibc-2.31-r0.apk

COPY . .

RUN pip install --no-cache-dir flask flask-cors \
    && pip install torch==2.0.0 --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

ENV HOST 0.0.0.0
ENV PORT 8080
EXPOSE 8080

CMD ["python", "main.py"]
