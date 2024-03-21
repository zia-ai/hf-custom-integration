# Work In Progress
FROM python:3.8.10

ENV POETRY_VERSION=1.3.2

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY poetry.lock pyproject.toml /src/

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-dev

COPY . /src/

ENTRYPOINT ["poetry", "run", "python3", "-m", "hf_integration.main"]
