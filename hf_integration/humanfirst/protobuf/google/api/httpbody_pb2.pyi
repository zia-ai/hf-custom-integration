"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Copyright 2019 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import builtins
import collections.abc
import google.protobuf.any_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class HttpBody(google.protobuf.message.Message):
    """Message that represents an arbitrary HTTP body. It should only be used for
    payload formats that can't be represented as JSON, such as raw binary or
    an HTML page.


    This message can be used both in streaming and non-streaming API methods in
    the request as well as the response.

    It can be used as a top-level request field, which is convenient if one
    wants to extract parameters from either the URL or HTTP template into the
    request fields and also want access to the raw HTTP body.

    Example:

        message GetResourceRequest {
          // A unique request id.
          string request_id = 1;

          // The raw HTTP body is bound to this field.
          google.api.HttpBody http_body = 2;
        }

        service ResourceService {
          rpc GetResource(GetResourceRequest) returns (google.api.HttpBody);
          rpc UpdateResource(google.api.HttpBody) returns
          (google.protobuf.Empty);
        }

    Example with streaming methods:

        service CaldavService {
          rpc GetCalendar(stream google.api.HttpBody)
            returns (stream google.api.HttpBody);
          rpc UpdateCalendar(stream google.api.HttpBody)
            returns (stream google.api.HttpBody);
        }

    Use of this type only changes how the request and response bodies are
    handled, all other features will continue to work unchanged.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CONTENT_TYPE_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    EXTENSIONS_FIELD_NUMBER: builtins.int
    content_type: builtins.str
    """The HTTP Content-Type header value specifying the content type of the body."""
    data: builtins.bytes
    """The HTTP request/response body as raw binary."""
    @property
    def extensions(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[google.protobuf.any_pb2.Any]:
        """Application specific response metadata. Must be set in the first response
        for streaming APIs.
        """
    def __init__(
        self,
        *,
        content_type: builtins.str = ...,
        data: builtins.bytes = ...,
        extensions: collections.abc.Iterable[google.protobuf.any_pb2.Any] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["content_type", b"content_type", "data", b"data", "extensions", b"extensions"]) -> None: ...

global___HttpBody = HttpBody
