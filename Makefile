PYTHON=python3
PROTOC=$(PYTHON) -m grpc_tools.protoc
PROTO_FILES=external_integration/v1alpha1/models.proto \
    external_integration/v1alpha1/discovery.proto \
    external_integration/v1alpha1/workspace.proto \
    external_nlu/v1alpha1/service.proto \
    playbook/data/config/v1alpha1/config.proto \
    playbook/entity.proto \
    playbook/span.proto \
    tags/tags.proto \
    longrunning/v1alpha1/operations.proto \
    model/core.proto \
    google/api/annotations.proto \
    google/api/http.proto \
    google/api/httpbody.proto

GIT_REV=$(shell git rev-parse --short @ | cut -c -7)

docker:
	docker build -t gcr.io/trial-184203/nlu-huggingface:$(GIT_REV) .

docker-gpu:
	docker build -t gcr.io/trial-184203/nlu-huggingface:$(GIT_REV)-gpu -f Dockerfile.gpu .

protoc:
	cd hf_integration/humanfirst/protobuf && $(PROTOC) \
		-I../../../../../spec \
		-I../../../../../vendor-spec \
		--python_out=. \
		--grpc_python_out=. \
		--mypy_out=. \
		$(PROTO_FILES)

tests:
	$(PYTHON) -m unittest
