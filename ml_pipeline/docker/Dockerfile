# Stage 1: Build
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Stage 2: Copy code and run
FROM base as runtime
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["uvicorn", "ml_pipeline.src.deployment.inference_server:app", "--host", "0.0.0.0", "--port", "8000"] 
