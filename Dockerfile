FROM python:3.10-slim

WORKDIR /app

COPY server.py .
COPY pyproject.toml .
COPY README.md .

RUN pip install --no-cache-dir mcp[cli] pyproj shapely httpx

ENTRYPOINT ["python", "server.py"]
