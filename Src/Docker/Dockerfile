FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Clone repos
RUN git clone https://github.com/your-org/repo1.git \
 && git clone https://github.com/your-org/repo2.git

# Copy local app code
COPY app/ ./app/
COPY start.sh .

# Make start.sh executable
RUN chmod +x start.sh

# Default command (can be overridden)
ENTRYPOINT ["./start.sh"]