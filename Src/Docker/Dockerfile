FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace
RUN git clone https://github.com/Tommycik/tesiControlNetFlux
RUN pip3 install git+https://github.com/huggingface/diffusers \
Run cd tesiControlNetFlux/Src
# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements_flux.txt && pip3 install bitsandbytes && pip3 install --upgrade diffusers && pip3 install peft

# Copy local app code
COPY start.sh .

# Make start.sh executable
RUN chmod +x start.sh

# Default command (can be overridden)
ENTRYPOINT ["./start.sh"]