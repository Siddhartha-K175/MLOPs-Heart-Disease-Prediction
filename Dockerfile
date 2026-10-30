FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pinned from uv.lock (core + Streamlit), so the image runs the same scikit-learn the
# model was trained with. The app imports src/ from /app, so the project itself is
# not pip-installed.
COPY website/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY .streamlit ./.streamlit
COPY api.py app.py db.py schema.sql ./
COPY artifacts/heart_model.joblib ./artifacts/heart_model.joblib

EXPOSE 8501 8000
# Hosts such as Render pass the port to listen on in $PORT; default to 8501 locally.
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
