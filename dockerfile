# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    build-essential \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python scripts and Galaxy XML wrappers to the container
COPY ./embedding_tool/figure_embedding_generator.py /app/embedding_splitter.py
COPY ./MIL_tool/mil_bag_creation.py /app/mil_bag_creator.py
COPY ./MIL_tool/mil_bag_creation.xml /app/galaxy_wrappers/
COPY ./embedding_tool/figure_embedding_generator.xml /app/galaxy_wrappers/

# Add a script to list available tools (optional)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose relevant ports (if necessary, e.g., for Galaxy integration)
EXPOSE 8080

# Set default command
ENTRYPOINT ["/app/entrypoint.sh"]
