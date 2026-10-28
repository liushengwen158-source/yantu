FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN pip install --no-cache-dir torch==2.4.0
WORKDIR /workspace
CMD ["python", "train.py"]
