"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.field_mask_pb2
import google.protobuf.internal.containers
import google.protobuf.message
import google.protobuf.timestamp_pb2
import google.protobuf.wrappers_pb2
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Tag(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    DESCRIPTION_FIELD_NUMBER: builtins.int
    COLOR_FIELD_NUMBER: builtins.int
    PROTECTED_FIELD_NUMBER: builtins.int
    PROTECTED_RECURSIVE_FIELD_NUMBER: builtins.int
    SOURCE_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    UPDATED_AT_FIELD_NUMBER: builtins.int
    DELETED_AT_FIELD_NUMBER: builtins.int
    id: builtins.str
    """Unique identifier of the tag."""
    name: builtins.str
    """Name of the tag."""
    description: builtins.str
    """Description of the tag."""
    color: builtins.str
    """Color of the tag in the UI.
    Any CSS color format is supported (ex: #FF0000, red).
    """
    protected: builtins.bool
    """Indicates that any object with this tag cannot be modified without explicit confirmation."""
    protected_recursive: builtins.bool
    """Indicates that any object with this tag, and their children cannot be modified without explicit confirmation.
    Ex: intent tagged with `protected_recursive` will also have their phrases protected
    """
    @property
    def source(self) -> global___TagSource:
        """Information on the source of the tag."""
    @property
    def created_at(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Creation date of the tag"""
    @property
    def updated_at(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """(Optional) Update date of the tag"""
    @property
    def deleted_at(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """(Optional) Deletion date of the tag"""
    def __init__(
        self,
        *,
        id: builtins.str = ...,
        name: builtins.str = ...,
        description: builtins.str = ...,
        color: builtins.str = ...,
        protected: builtins.bool = ...,
        protected_recursive: builtins.bool = ...,
        source: global___TagSource | None = ...,
        created_at: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        updated_at: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        deleted_at: google.protobuf.timestamp_pb2.Timestamp | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["created_at", b"created_at", "deleted_at", b"deleted_at", "source", b"source", "updated_at", b"updated_at"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["color", b"color", "created_at", b"created_at", "deleted_at", b"deleted_at", "description", b"description", "id", b"id", "name", b"name", "protected", b"protected", "protected_recursive", b"protected_recursive", "source", b"source", "updated_at", b"updated_at"]) -> None: ...

global___Tag = Tag

class TagSource(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class DialogflowCx(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        FLOW_ID_FIELD_NUMBER: builtins.int
        PAGE_ID_FIELD_NUMBER: builtins.int
        TRANSITION_ROUTE_GROUP_ID_FIELD_NUMBER: builtins.int
        flow_id: builtins.str
        """If defined, indicates that the tag was created to correspond to a DialogFlow flow.
        This is used to properly re-assign intents to their corresponding flows based on tags.
        """
        page_id: builtins.str
        """If defined, indicates that the tag was created to correspond to a DialogFlow page.
        This is used to properly re-assign intents to their corresponding pages based on tags.
        """
        transition_route_group_id: builtins.str
        """If defined, indicates that the tag was created to correspond to a DialogFlow transition route group.
        This is used to properly re-assign intents to their corresponding transition route group based on tags.
        """
        def __init__(
            self,
            *,
            flow_id: builtins.str = ...,
            page_id: builtins.str = ...,
            transition_route_group_id: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["flow_id", b"flow_id", "page_id", b"page_id", "transition_route_group_id", b"transition_route_group_id"]) -> None: ...

    class DialogflowEs(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        class Priority(google.protobuf.message.Message):
            """Indicates that the tag was created to correspond to a DialogFlow intent priority.
            This is used to properly re-assign default priority values to intents based on tags.
            """

            DESCRIPTOR: google.protobuf.descriptor.Descriptor

            VALUE_FIELD_NUMBER: builtins.int
            @property
            def value(self) -> google.protobuf.wrappers_pb2.Int32Value: ...
            def __init__(
                self,
                *,
                value: google.protobuf.wrappers_pb2.Int32Value | None = ...,
            ) -> None: ...
            def HasField(self, field_name: typing_extensions.Literal["value", b"value"]) -> builtins.bool: ...
            def ClearField(self, field_name: typing_extensions.Literal["value", b"value"]) -> None: ...

        class TopLevel(google.protobuf.message.Message):
            """Indicates that the tag was created to correspond to a DialogFlow intent that is a top level intent."""

            DESCRIPTOR: google.protobuf.descriptor.Descriptor

            def __init__(
                self,
            ) -> None: ...

        class Context(google.protobuf.message.Message):
            """Indicates that the tag was created to correspond to a DialogFlow context that is attached to an intent."""

            DESCRIPTOR: google.protobuf.descriptor.Descriptor

            NAME_FIELD_NUMBER: builtins.int
            name: builtins.str
            def __init__(
                self,
                *,
                name: builtins.str = ...,
            ) -> None: ...
            def ClearField(self, field_name: typing_extensions.Literal["name", b"name"]) -> None: ...

        class FollowUp(google.protobuf.message.Message):
            """Indicates that the tag was created to correspond to a DialogFlow follow up intent."""

            DESCRIPTOR: google.protobuf.descriptor.Descriptor

            def __init__(
                self,
            ) -> None: ...

        PRIORITY_FIELD_NUMBER: builtins.int
        TOP_LEVEL_FIELD_NUMBER: builtins.int
        CONTEXT_FIELD_NUMBER: builtins.int
        FOLLOW_UP_FIELD_NUMBER: builtins.int
        @property
        def priority(self) -> global___TagSource.DialogflowEs.Priority: ...
        @property
        def top_level(self) -> global___TagSource.DialogflowEs.TopLevel: ...
        @property
        def context(self) -> global___TagSource.DialogflowEs.Context: ...
        @property
        def follow_up(self) -> global___TagSource.DialogflowEs.FollowUp: ...
        def __init__(
            self,
            *,
            priority: global___TagSource.DialogflowEs.Priority | None = ...,
            top_level: global___TagSource.DialogflowEs.TopLevel | None = ...,
            context: global___TagSource.DialogflowEs.Context | None = ...,
            follow_up: global___TagSource.DialogflowEs.FollowUp | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["context", b"context", "follow_up", b"follow_up", "priority", b"priority", "top_level", b"top_level", "type", b"type"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["context", b"context", "follow_up", b"follow_up", "priority", b"priority", "top_level", b"top_level", "type", b"type"]) -> None: ...
        def WhichOneof(self, oneof_group: typing_extensions.Literal["type", b"type"]) -> typing_extensions.Literal["priority", "top_level", "context", "follow_up"] | None: ...

    SOURCE_ID_FIELD_NUMBER: builtins.int
    MERGED_IDS_FIELD_NUMBER: builtins.int
    DIALOGFLOW_CX_FIELD_NUMBER: builtins.int
    DIALOGFLOW_ES_FIELD_NUMBER: builtins.int
    source_id: builtins.str
    """ID of the tag at its source if it has been imported (if applicable)"""
    @property
    def merged_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """List of unique identifiers of tags from HumanFirst or
        external integrations that got merged into this tag
        at some point and that can be reused to ease further
        merges.

        This list may not be exhaustive and could be truncated.
        """
    @property
    def dialogflow_cx(self) -> global___TagSource.DialogflowCx: ...
    @property
    def dialogflow_es(self) -> global___TagSource.DialogflowEs: ...
    def __init__(
        self,
        *,
        source_id: builtins.str = ...,
        merged_ids: collections.abc.Iterable[builtins.str] | None = ...,
        dialogflow_cx: global___TagSource.DialogflowCx | None = ...,
        dialogflow_es: global___TagSource.DialogflowEs | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["dialogflow_cx", b"dialogflow_cx", "dialogflow_es", b"dialogflow_es", "type", b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["dialogflow_cx", b"dialogflow_cx", "dialogflow_es", b"dialogflow_es", "merged_ids", b"merged_ids", "source_id", b"source_id", "type", b"type"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["type", b"type"]) -> typing_extensions.Literal["dialogflow_cx", "dialogflow_es"] | None: ...

global___TagSource = TagSource

class TagReference(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    PROTECTED_FIELD_NUMBER: builtins.int
    id: builtins.str
    """Unique identifier of the tag."""
    name: builtins.str
    """(Optional) Only used when importing data that tag IDs are not defined yet.
    This will not be filled when requesting tagged objects.
    """
    protected: builtins.bool
    """For internal use. There is no guarantee that this will be properly filled."""
    def __init__(
        self,
        *,
        id: builtins.str = ...,
        name: builtins.str = ...,
        protected: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["id", b"id", "name", b"name", "protected", b"protected"]) -> None: ...

global___TagReference = TagReference

class TagReferences(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TAGS_FIELD_NUMBER: builtins.int
    @property
    def tags(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TagReference]: ...
    def __init__(
        self,
        *,
        tags: collections.abc.Iterable[global___TagReference] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tags", b"tags"]) -> None: ...

global___TagReferences = TagReferences

class TagPredicate(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    REQUIRE_IDS_FIELD_NUMBER: builtins.int
    INCLUDE_IDS_FIELD_NUMBER: builtins.int
    EXCLUDE_IDS_FIELD_NUMBER: builtins.int
    @property
    def require_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Only include objects with ALL of the given tag ids."""
    @property
    def include_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Only include objects with ANY of the given tag ids."""
    @property
    def exclude_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Exclude objects with ANY of the given tag ids."""
    def __init__(
        self,
        *,
        require_ids: collections.abc.Iterable[builtins.str] | None = ...,
        include_ids: collections.abc.Iterable[builtins.str] | None = ...,
        exclude_ids: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["exclude_ids", b"exclude_ids", "include_ids", b"include_ids", "require_ids", b"require_ids"]) -> None: ...

global___TagPredicate = TagPredicate

class CreateTagRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    PLAYBOOK_ID_FIELD_NUMBER: builtins.int
    TAG_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the playbook"""
    playbook_id: builtins.str
    """Metastore ID of the playbook"""
    @property
    def tag(self) -> global___Tag: ...
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        playbook_id: builtins.str = ...,
        tag: global___Tag | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["tag", b"tag"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["namespace", b"namespace", "playbook_id", b"playbook_id", "tag", b"tag"]) -> None: ...

global___CreateTagRequest = CreateTagRequest

class CreateTagResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TAG_FIELD_NUMBER: builtins.int
    @property
    def tag(self) -> global___Tag: ...
    def __init__(
        self,
        *,
        tag: global___Tag | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["tag", b"tag"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["tag", b"tag"]) -> None: ...

global___CreateTagResponse = CreateTagResponse

class ListTagsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    PLAYBOOK_ID_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the playbook"""
    playbook_id: builtins.str
    """Metastore ID of the playbook"""
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        playbook_id: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["namespace", b"namespace", "playbook_id", b"playbook_id"]) -> None: ...

global___ListTagsRequest = ListTagsRequest

class ListTagsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TAGS_FIELD_NUMBER: builtins.int
    @property
    def tags(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Tag]: ...
    def __init__(
        self,
        *,
        tags: collections.abc.Iterable[global___Tag] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tags", b"tags"]) -> None: ...

global___ListTagsResponse = ListTagsResponse

class UpdateTagRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    PLAYBOOK_ID_FIELD_NUMBER: builtins.int
    TAG_FIELD_NUMBER: builtins.int
    UPDATE_MASK_FIELD_NUMBER: builtins.int
    UPDATE_ALL_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the playbook"""
    playbook_id: builtins.str
    """Metastore ID of the playbook"""
    @property
    def tag(self) -> global___Tag:
        """Tag to update"""
    @property
    def update_mask(self) -> google.protobuf.field_mask_pb2.FieldMask:
        """List of fields to be updated."""
    update_all: builtins.bool
    """If no mask specified, indicates that all fields need to be updated."""
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        playbook_id: builtins.str = ...,
        tag: global___Tag | None = ...,
        update_mask: google.protobuf.field_mask_pb2.FieldMask | None = ...,
        update_all: builtins.bool = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["tag", b"tag", "update_mask", b"update_mask"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["namespace", b"namespace", "playbook_id", b"playbook_id", "tag", b"tag", "update_all", b"update_all", "update_mask", b"update_mask"]) -> None: ...

global___UpdateTagRequest = UpdateTagRequest

class UpdateTagResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TAG_FIELD_NUMBER: builtins.int
    @property
    def tag(self) -> global___Tag: ...
    def __init__(
        self,
        *,
        tag: global___Tag | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["tag", b"tag"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["tag", b"tag"]) -> None: ...

global___UpdateTagResponse = UpdateTagResponse

class DeleteTagRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAMESPACE_FIELD_NUMBER: builtins.int
    PLAYBOOK_ID_FIELD_NUMBER: builtins.int
    TAG_ID_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    """Namespace of the playbook"""
    playbook_id: builtins.str
    """Metastore ID of the playbook"""
    tag_id: builtins.str
    """ID of the tag to delete"""
    def __init__(
        self,
        *,
        namespace: builtins.str = ...,
        playbook_id: builtins.str = ...,
        tag_id: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["namespace", b"namespace", "playbook_id", b"playbook_id", "tag_id", b"tag_id"]) -> None: ...

global___DeleteTagRequest = DeleteTagRequest

class DeleteTagResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___DeleteTagResponse = DeleteTagResponse
