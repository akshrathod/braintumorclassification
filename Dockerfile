FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only service and minimal deps for serving
COPY service ./service
COPY app/utils.py app/gradcam.py ./app/
COPY artifacts ./artifacts

ENV FLASK_APP=service/app.py
EXPOSE 5000
CMD ["python", "-m", "flask", "run", "-p", "5000", "--host", "0.0.0.0"]
