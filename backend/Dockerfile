# syntax=docker/dockerfile:1

# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install uv for fast dependency resolution
RUN pip install uv

# Copy requirements/pyproject files
COPY requirements.txt pyproject.toml ./

# Install dependencies into the system python (since we're in a container)
RUN uv pip install --system -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY ./app /app/app
COPY ./scripts /app/scripts

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Command to run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
