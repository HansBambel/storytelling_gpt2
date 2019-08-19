FROM pytorch/pytorch:latest
RUN pip install pytorch_transformers

# Set the working directory to /app
WORKDIR /gpt2

# Copy the current directory contents into the container at /app
# COPY . /gpt2