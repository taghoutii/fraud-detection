FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime, which the slim image doesn't ship
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# serving-only dependencies (not the full training requirements.txt), installed first for layer caching
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY app/ app/
COPY src/ src/
COPY models/final_pipeline.joblib models/
# only the threshold file is needed at runtime; .dockerignore keeps the other results out
COPY reports/results/ reports/results/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
