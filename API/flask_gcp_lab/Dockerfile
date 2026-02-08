FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source files
COPY src/ ./src/

# Train model (this creates model/ directory)
RUN PYTHONPATH=/app python src/train.py

# This line tells Docker that the container will listen on the port specified by the environment variable PORT
EXPOSE $PORT 

# Add /app to PYTHONPATH so imports work
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python src/main.py"]
# sh to allow for accessing environmental variables like $PORT
# -c to run this within the container