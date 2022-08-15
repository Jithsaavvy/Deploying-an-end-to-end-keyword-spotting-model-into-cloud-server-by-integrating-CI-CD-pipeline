FROM python:3.8

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      --no-install-recommends \
      libsndfile1 \
      libsndfile1-dev \
    && :

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE $PORT

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
