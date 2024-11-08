FROM python:3.8.10

ENV POETRY_VERSION=1.3.2

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src

# Clone the repository from GitHub
# Clone from a particular branch
# RUN git clone -b <branch_name> --single-branch https://github.com/zia-ai/hf-custom-integration.git .

# Clone from master branch
RUN git clone https://github.com/zia-ai/hf-custom-integration.git .

# Generate MTLS credentials from the commands given in the README.md
COPY ./credentials /src/credentials

RUN poetry config virtualenvs.create false && poetry install
