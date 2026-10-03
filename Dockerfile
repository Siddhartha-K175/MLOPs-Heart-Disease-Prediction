FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY api.py app.py ./
COPY artifacts ./artifacts

EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
