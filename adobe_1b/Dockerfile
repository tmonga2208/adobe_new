# Dockerfile for the PDF Ingestion and Querying System

# Use a lightweight Python base image
# Python 3.9 is chosen for broad compatibility, slim-buster for smaller image size.
FROM python:3.9-slim-buster

# Set environment variables
# PYTHONUNBUFFERED ensures Python output is not buffered, useful for logging.
# APP_HOME defines the working directory inside the container.
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies required by PyMuPDF (fitz)
# build-essential: For compiling some Python packages if they have C extensions.
# libgl1-mesa-glx: Provides OpenGL libraries, sometimes needed by rendering-related Python packages like PyMuPDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    # Clean up apt caches to reduce the final image size
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container and install Python dependencies.
# --no-cache-dir: Prevents pip from storing cached downloads, reducing image size.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model to ensure it's available offline at runtime.
# This step makes the Docker image slightly larger but guarantees the model is present
# without requiring internet access when the container is run.
# The 'all-MiniLM-L6-v2' model is approximately 90MB, well within the 1GB model size constraint.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy all your application code (Python scripts) into the container's working directory.
# Ensure your Python scripts (e.g., ingestion_pipeline.py, query_system.py)
# are in the same directory as this Dockerfile.
COPY . $APP_HOME/

# Create directories inside the container for input PDFs and ChromaDB persistence.
# These directories will be used by your Python scripts.
RUN mkdir -p ${APP_HOME}/input_pdfs
RUN mkdir -p ${APP_HOME}/chroma_data

# Define the default command to run when the container starts.
# This allows you to specify which script to run (e.g., ingestion or query)
# when you execute the 'docker run' command.
# For example:
# To run the ingestion pipeline:
#   docker run -v /path/to/your/host/input_pdfs:/app/input_pdfs -v /path/to/your/host/chroma_data:/app/chroma_data my-rag-app python ingestion_pipeline.py
# To run the query system:
#   docker run -v /path/to/your/host/chroma_data:/app/chroma_data my-rag-app python query_system.py
CMD ["python"]