# =============================================================================
#  Unstitch backend - Hugging Face Spaces (Docker SDK)
# =============================================================================
#  Why a container and not a serverless function: the API answers 202 and keeps
#  working after the response is sent. A Vercel/Lambda-style function is killed
#  when the request ends, so the polling job model this app is built on cannot
#  run there. See docs/PLAN.md section 10.
# =============================================================================
FROM python:3.11-slim

# 3.11 matches the local venv exactly, so "works on my machine" and "works in
# the Space" are the same claim. (Verified: every dependency, including
# scenedetect and OpenCV 5, also installs cleanly on 3.14 - the pin is for
# dev/prod parity, not a wheel gap.)

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg          - every render, cut, thumbnail and probe in the app
# libgl1,
# libglib2.0-0    - shared libraries the opencv-python wheel links against.
#                   Needed because scenedetect 0.7 requires the GUI build of
#                   OpenCV; see the note in requirements.txt. Two apt packages
#                   is cheaper than shipping two copies of cv2.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as UID 1000. Writing the workspace as a
# non-root user we create ourselves keeps local `docker run` and the Space
# behaving identically.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies before source: editing a Python file must not invalidate the
# ~400MB OpenCV layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# MEDIA_ROOT defaults to ./work, resolved relative to this directory.
RUN mkdir -p /app/work && chown -R app:app /app
USER app

# 7860 is the port Spaces routes to by default.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
