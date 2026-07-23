FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DASH_HOST=0.0.0.0 \
    DASH_DEBUG=false \
    PORT=8050

WORKDIR /app

COPY requirements.txt requirements-deploy.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-deploy.txt \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser dashboard ./dashboard
COPY --chown=appuser:appuser data/processed ./data/processed

USER appuser

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8050') + '/', timeout=4)"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 dashboard.wsgi:application"]
