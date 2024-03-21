"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.timestamp_pb2
import longrunning.v1alpha1.operations_pb2
import model.core_pb2
import playbook.data.config.v1alpha1.config_pb2
import playbook.training_phrase_pb2
import sys
import tags.tags_pb2
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class ExportIntentsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    PLAYBOOK_ID_FIELD_NUMBER: builtins.int
    FORMAT_FIELD_NUMBER: builtins.int
    FORMAT_OPTIONS_FIELD_NUMBER: builtins.int
    INTENT_IDS_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the workspace."""
    playbook_id: builtins.str
    """Unique identifier of the workspace."""
    format: playbook.data.config.v1alpha1.config_pb2.IntentsDataFormat.ValueType
    """Format of the exported data.

    Values:
    1 = CSV
    2 = Rasa 1 Markdown
    3 = Rasa 2 YAML
    4 = Botpress
    6 = Dialogflow ES
    7 = Humanfirst JSON
    """
    @property
    def format_options(self) -> playbook.data.config.v1alpha1.config_pb2.IntentsDataOptions:
        """Options of the format of the exported data."""
    @property
    def intent_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """(Optional) Limit export to these given intents."""
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        playbook_id: builtins.str = ...,
        format: playbook.data.config.v1alpha1.config_pb2.IntentsDataFormat.ValueType = ...,
        format_options: playbook.data.config.v1alpha1.config_pb2.IntentsDataOptions | None = ...,
        intent_ids: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["format_options", b"format_options"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["format", b"format", "format_options", b"format_options", "intent_ids", b"intent_ids", "namespace", b"namespace", "playbook_id", b"playbook_id"]) -> None: ...

global___ExportIntentsRequest = ExportIntentsRequest

class ExportIntentsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATA_FIELD_NUMBER: builtins.int
    PROBLEMS_FIELD_NUMBER: builtins.int
    TOTAL_PROBLEMS_FIELD_NUMBER: builtins.int
    data: builtins.bytes
    """Bytes of the exported file.
    The format is the one requested through the `format` field in request.
    """
    @property
    def problems(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ValidationProblem]:
        """List of problems that occurred while exporting intents.
        This list may not be exhaustive. If it's been limited, the `total_problems` indicate the total count.
        """
    total_problems: builtins.int
    """Indicates total number of problems at import."""
    def __init__(
        self,
        *,
        data: builtins.bytes = ...,
        problems: collections.abc.Iterable[global___ValidationProblem] | None = ...,
        total_problems: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["data", b"data", "problems", b"problems", "total_problems", b"total_problems"]) -> None: ...

global___ExportIntentsResponse = ExportIntentsResponse

class ImportIntentsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    PLAYBOOK_ID_FIELD_NUMBER: builtins.int
    FORMAT_FIELD_NUMBER: builtins.int
    FORMAT_OPTIONS_FIELD_NUMBER: builtins.int
    IMPORT_OPTIONS_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    CLEAR_WORKSPACE_FIELD_NUMBER: builtins.int
    CLEAR_INTENTS_FIELD_NUMBER: builtins.int
    CLEAR_ENTITIES_FIELD_NUMBER: builtins.int
    CLEAR_TAGS_FIELD_NUMBER: builtins.int
    MERGE_INTENTS_FIELD_NUMBER: builtins.int
    MERGE_ENTITIES_FIELD_NUMBER: builtins.int
    MERGE_TAGS_FIELD_NUMBER: builtins.int
    SOFT_FAIL_FIELD_NUMBER: builtins.int
    EXTRA_INTENT_TAGS_FIELD_NUMBER: builtins.int
    EXTRA_PHRASE_TAGS_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the workspace."""
    playbook_id: builtins.str
    """Unique identifier of the workspace."""
    format: playbook.data.config.v1alpha1.config_pb2.IntentsDataFormat.ValueType
    """Format of the imported file.

    Values:
    1 = CSV
    2 = Rasa 1 Markdown
    3 = Rasa 2 YAML
    4 = Botpress
    6 = Dialogflow ES
    7 = Humanfirst JSON
    """
    @property
    def format_options(self) -> playbook.data.config.v1alpha1.config_pb2.IntentsDataOptions:
        """Optional options of the format of the imported data."""
    @property
    def import_options(self) -> playbook.data.config.v1alpha1.config_pb2.ImportOptions:
        """Optional options for importation (merge, clear, etc.)
        If specified, the deprecated `*_clear`, `*_merge` and `extra_*_tags` fields will be ignored.
        """
    data: builtins.bytes
    """Bytes of the file to import.
    The format is the one requested through the `format` field in request.
    """
    clear_workspace: builtins.bool
    """Clears workspace intents, entities & tags before importing.
    Deprecated: use `import_options`
    """
    clear_intents: builtins.bool
    """Clears workspace intents before importing.
    Deprecated: use `import_options`
    """
    clear_entities: builtins.bool
    """Clears workspace entities before importing.
    Deprecated: use `import_options`
    """
    clear_tags: builtins.bool
    """Clears workspace tags before importing.
    Note: should not be used in combination with `extra_intent_tags` or `extra_phrase_tags` since
          we will clear potentially referenced tags.
    Deprecated: use `import_options`
    """
    merge_intents: builtins.bool
    """Tries to merge intents into existing ones if they can be found in the workspace.
    Deprecated: use `import_options`
    """
    merge_entities: builtins.bool
    """Tries to merge entities into existing ones if they can be found in the workspace.
    Deprecated: use `import_options`
    """
    merge_tags: builtins.bool
    """Tries to merge tags into existing ones if they can be found in the workspace.
    Deprecated: use `import_options`
    """
    soft_fail: builtins.bool
    """Returns fatal problems via the `problems` field instead of gRPC errors.
    Temporary flag until front-end properly handles soft failures instead of gRPC error.
    """
    @property
    def extra_intent_tags(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tags.tags_pb2.TagReference]:
        """Add extra tags to imported intents.
        Deprecated: use `import_options`
        """
    @property
    def extra_phrase_tags(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tags.tags_pb2.TagReference]:
        """Add extra tags to imported phrases.
        Deprecated: use `import_options`
        """
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        playbook_id: builtins.str = ...,
        format: playbook.data.config.v1alpha1.config_pb2.IntentsDataFormat.ValueType = ...,
        format_options: playbook.data.config.v1alpha1.config_pb2.IntentsDataOptions | None = ...,
        import_options: playbook.data.config.v1alpha1.config_pb2.ImportOptions | None = ...,
        data: builtins.bytes = ...,
        clear_workspace: builtins.bool = ...,
        clear_intents: builtins.bool = ...,
        clear_entities: builtins.bool = ...,
        clear_tags: builtins.bool = ...,
        merge_intents: builtins.bool = ...,
        merge_entities: builtins.bool = ...,
        merge_tags: builtins.bool = ...,
        soft_fail: builtins.bool = ...,
        extra_intent_tags: collections.abc.Iterable[tags.tags_pb2.TagReference] | None = ...,
        extra_phrase_tags: collections.abc.Iterable[tags.tags_pb2.TagReference] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["format_options", b"format_options", "import_options", b"import_options"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["clear_entities", b"clear_entities", "clear_intents", b"clear_intents", "clear_tags", b"clear_tags", "clear_workspace", b"clear_workspace", "data", b"data", "extra_intent_tags", b"extra_intent_tags", "extra_phrase_tags", b"extra_phrase_tags", "format", b"format", "format_options", b"format_options", "import_options", b"import_options", "merge_entities", b"merge_entities", "merge_intents", b"merge_intents", "merge_tags", b"merge_tags", "namespace", b"namespace", "playbook_id", b"playbook_id", "soft_fail", b"soft_fail"]) -> None: ...

global___ImportIntentsRequest = ImportIntentsRequest

class ImportIntentsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    IMPORTED_INTENT_COUNT_FIELD_NUMBER: builtins.int
    IMPORTED_TRAINING_PHRASE_COUNT_FIELD_NUMBER: builtins.int
    PROBLEMS_FIELD_NUMBER: builtins.int
    TOTAL_PROBLEMS_FIELD_NUMBER: builtins.int
    BACKGROUND_OPERATION_FIELD_NUMBER: builtins.int
    imported_intent_count: builtins.int
    """Number of intents that were imported."""
    imported_training_phrase_count: builtins.int
    """Number of training phrases that were imported."""
    @property
    def problems(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ValidationProblem]:
        """List of problems that occurred while validating the file format.
        This list may not be exhaustive. If it's been limited, the `total_problems` indicate the total count.
        """
    total_problems: builtins.int
    """Indicates total number of problems at import."""
    @property
    def background_operation(self) -> longrunning.v1alpha1.operations_pb2.Operation:
        """If this is set, the importation was not completed yet.
        The user should poll `GetOperation` until it's done.
        """
    def __init__(
        self,
        *,
        imported_intent_count: builtins.int = ...,
        imported_training_phrase_count: builtins.int = ...,
        problems: collections.abc.Iterable[global___ValidationProblem] | None = ...,
        total_problems: builtins.int = ...,
        background_operation: longrunning.v1alpha1.operations_pb2.Operation | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["background_operation", b"background_operation"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["background_operation", b"background_operation", "imported_intent_count", b"imported_intent_count", "imported_training_phrase_count", b"imported_training_phrase_count", "problems", b"problems", "total_problems", b"total_problems"]) -> None: ...

global___ImportIntentsResponse = ImportIntentsResponse

class ImportConversationsFileRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    CONVERSATION_SOURCE_ID_FIELD_NUMBER: builtins.int
    FILENAME_FIELD_NUMBER: builtins.int
    FORMAT_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    SOFT_FAIL_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the workspace."""
    conversation_source_id: builtins.str
    """Unique identifier of the conversation source in which we want to import the file."""
    filename: builtins.str
    """Name of the file that is being uploaded.
    Note: this name may be mangled by the backend if it contains invalid characters. The mangled name will be
    returned in the response.
    """
    format: model.core_pb2.ConversationsImportFormat.ValueType
    """File format."""
    data: builtins.bytes
    """File content."""
    soft_fail: builtins.bool
    """Returns fatal problems via the `problems` field instead of gRPC errors.
    Temporary flag until front-end properly handles soft failures instead of gRPC error.
    """
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        conversation_source_id: builtins.str = ...,
        filename: builtins.str = ...,
        format: model.core_pb2.ConversationsImportFormat.ValueType = ...,
        data: builtins.bytes = ...,
        soft_fail: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["conversation_source_id", b"conversation_source_id", "data", b"data", "filename", b"filename", "format", b"format", "namespace", b"namespace", "soft_fail", b"soft_fail"]) -> None: ...

global___ImportConversationsFileRequest = ImportConversationsFileRequest

class ImportConversationsFileResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FILENAME_FIELD_NUMBER: builtins.int
    PROBLEMS_FIELD_NUMBER: builtins.int
    TOTAL_PROBLEMS_FIELD_NUMBER: builtins.int
    filename: builtins.str
    """File name as saved in the user upload location as mangled if invalid characters were present in it."""
    @property
    def problems(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ValidationProblem]:
        """List of problems that occurred while validating the file format.
        This list may not be exhaustive. If it's been limited, the `total_problems` indicate the total count.
        """
    total_problems: builtins.int
    """Indicates total number of problems at import."""
    def __init__(
        self,
        *,
        filename: builtins.str = ...,
        problems: collections.abc.Iterable[global___ValidationProblem] | None = ...,
        total_problems: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["filename", b"filename", "problems", b"problems", "total_problems", b"total_problems"]) -> None: ...

global___ImportConversationsFileResponse = ImportConversationsFileResponse

class ListConversationsFileRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    CONVERSATION_SOURCE_ID_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the workspace."""
    conversation_source_id: builtins.str
    """Unique identifier of the conversation source in which we want to import the file."""
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        conversation_source_id: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["conversation_source_id", b"conversation_source_id", "namespace", b"namespace"]) -> None: ...

global___ListConversationsFileRequest = ListConversationsFileRequest

class ListConversationsFileResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class File(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        NAME_FIELD_NUMBER: builtins.int
        FORMAT_FIELD_NUMBER: builtins.int
        UPLOAD_TIME_FIELD_NUMBER: builtins.int
        FROM_LAST_UPLOAD_FIELD_NUMBER: builtins.int
        name: builtins.str
        """File name"""
        format: model.core_pb2.ConversationsImportFormat.ValueType
        """Format of the conversations in file"""
        @property
        def upload_time(self) -> google.protobuf.timestamp_pb2.Timestamp:
            """Indicates the time at which the file got uploaded"""
        from_last_upload: builtins.bool
        """Indicates that the file was part of the last upload"""
        def __init__(
            self,
            *,
            name: builtins.str = ...,
            format: model.core_pb2.ConversationsImportFormat.ValueType = ...,
            upload_time: google.protobuf.timestamp_pb2.Timestamp | None = ...,
            from_last_upload: builtins.bool = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["upload_time", b"upload_time"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["format", b"format", "from_last_upload", b"from_last_upload", "name", b"name", "upload_time", b"upload_time"]) -> None: ...

    FILES_FIELD_NUMBER: builtins.int
    @property
    def files(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ListConversationsFileResponse.File]:
        """Files"""
    def __init__(
        self,
        *,
        files: collections.abc.Iterable[global___ListConversationsFileResponse.File] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["files", b"files"]) -> None: ...

global___ListConversationsFileResponse = ListConversationsFileResponse

class DeleteConversationsFileRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    CONVERSATION_SOURCE_ID_FIELD_NUMBER: builtins.int
    FILENAME_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the workspace."""
    conversation_source_id: builtins.str
    """Unique identifier of the conversation source in which we want to import the file."""
    filename: builtins.str
    """Name of the file to delete."""
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        conversation_source_id: builtins.str = ...,
        filename: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["conversation_source_id", b"conversation_source_id", "filename", b"filename", "namespace", b"namespace"]) -> None: ...

global___DeleteConversationsFileRequest = DeleteConversationsFileRequest

class DeleteConversationsFileResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___DeleteConversationsFileResponse = DeleteConversationsFileResponse

class ValidationProblem(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _Level:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _LevelEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[ValidationProblem._Level.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        INVALID: ValidationProblem._Level.ValueType  # 0
        WARNING: ValidationProblem._Level.ValueType  # 1
        FATAL: ValidationProblem._Level.ValueType  # 2

    class Level(_Level, metaclass=_LevelEnumTypeWrapper): ...
    INVALID: ValidationProblem.Level.ValueType  # 0
    WARNING: ValidationProblem.Level.ValueType  # 1
    FATAL: ValidationProblem.Level.ValueType  # 2

    LEVEL_FIELD_NUMBER: builtins.int
    MESSAGE_FIELD_NUMBER: builtins.int
    FILENAME_FIELD_NUMBER: builtins.int
    LINE_FIELD_NUMBER: builtins.int
    TRAINING_PHRASE_FIELD_NUMBER: builtins.int
    level: global___ValidationProblem.Level.ValueType
    """Level of the problem.
    1 = Warning
    2 = Fatal
    """
    message: builtins.str
    """Message of the problem."""
    filename: builtins.str
    """(Optional) Filename in which the problem was encountered."""
    line: builtins.int
    """(Optional) Line of `filename` on which the problem was encountered."""
    @property
    def training_phrase(self) -> playbook.training_phrase_pb2.TrainingPhrase:
        """(Optional) Training phrase on which the problem was encountered."""
    def __init__(
        self,
        *,
        level: global___ValidationProblem.Level.ValueType = ...,
        message: builtins.str = ...,
        filename: builtins.str = ...,
        line: builtins.int = ...,
        training_phrase: playbook.training_phrase_pb2.TrainingPhrase | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["object", b"object", "training_phrase", b"training_phrase"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["filename", b"filename", "level", b"level", "line", b"line", "message", b"message", "object", b"object", "training_phrase", b"training_phrase"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["object", b"object"]) -> typing_extensions.Literal["training_phrase"] | None: ...

global___ValidationProblem = ValidationProblem
