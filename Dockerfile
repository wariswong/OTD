FROM python:3.11-slim

# libgomp1: OpenMP runtime some of the scientific-stack wheels (numpy/scipy/
# opendssdirect's bundled solvers) link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime data dirs the app writes to at request time (uploads, generated
# reports, shareload run output, logs) — not part of the image's content,
# just need to exist. Mount volumes over these in production so the data
# survives container restarts/redeploys.
RUN mkdir -p uploads output logs feature_shareload/FEASIBLE

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# --worker-class gthread --workers 1 --threads 4: the shareload background
# job tracker (_running_jobs in app/routes/shareload.py) is an in-memory
# Python set — it only stays correct with a single worker process. Multiple
# gunicorn *workers* would each have their own copy and no longer agree on
# which pairs are already running. Threads within the one worker still give
# concurrent request handling; scale further by moving that tracker to
# Redis/DB before adding workers.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--worker-class", "gthread", "--workers", "1", "--threads", "4", "--timeout", "120", "run:app"]
