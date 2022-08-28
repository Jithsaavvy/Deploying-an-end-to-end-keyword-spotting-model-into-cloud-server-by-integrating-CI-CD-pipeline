# Build base image
FROM python:3.8-slim as python-base

ENV PYTHONUNBUFFERED = 1 \
    PYTHONDONTWRITEBYTECODE = 1 \
    PIP_NO_CACHE_DIR = off \
    PIP_DISABLE_PIP_VERSION_CHECK = on \
    PIP_DEFAULT_TIMEOUT = 100 \
    POETRY_VERSION = 1.1.15 \
    POETRY_HOME = "/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT = true \
    POETRY_NO_INTERACTION = 1 \
    PYSETUP_PATH = "/opt/pysetup" \
    VENV_PATH = "/opt/pysetup/.venv"

ENV PATH = "$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# Build dev image
FROM python-base as dev-base

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      --no-install-recommends \
      curl \
      build-essential \
      libsndfile1 \
      libsndfile1-dev

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

ENV PATH="${PATH}:/root/.poetry/bin"

COPY poetry.lock pyproject.toml ./

RUN poetry install


#Build production image
FROM python-base as production

COPY --from=dev-base $PYSETUP_PATH $PYSETUP_PATH

COPY . /app

EXPOSE $PORT

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app