# The code sandbox service: the only container that holds the Docker socket.
# It can only start the fixed, locked-down container in server.py.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY server.py .
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/status', timeout=4)"
# Root inside this container only to use the mounted Docker socket (root:docker on the host).
CMD ["python", "-I", "server.py"]
