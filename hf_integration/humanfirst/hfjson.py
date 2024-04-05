from dataclasses import dataclass
from typing import Dict, Optional, Any, List, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


@dataclass
class EntitySource:
    """(Optional) Information about the source of the entity."""
    """(Optional) Extra metadata about the entity in the source system."""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) Unique identifier of the entity in the source system."""
    source_id: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'EntitySource':
        assert isinstance(obj, dict)
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        source_id = from_union([from_str, from_none], obj.get("source_id"))
        return EntitySource(metadata, source_id)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.source_id is not None:
            result["source_id"] = from_union([from_str, from_none], self.source_id)
        return result


@dataclass
class EntityTag:
    """(Optional) Unique identifier of the tag.
    If not specified, `name` needs to be specified.
    """
    id: Optional[str] = None
    """(Optional) Name of the tag.
    If not specified, `id` needs to be specified.
    """
    name: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'EntityTag':
        assert isinstance(obj, dict)
        id = from_union([from_str, from_none], obj.get("id"))
        name = from_union([from_str, from_none], obj.get("name"))
        return EntityTag(id, name)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        return result


@dataclass
class Span:
    """(Optional) Span of the entity in the example.
    If the reference is provided through a `TextPart`, the span is not required
    since parts are presenting the text and entities.
    """
    """Inclusive start index of the span in characters."""
    from_character: int
    """Exclusive end index of the span in characters."""
    to_character: int

    @staticmethod
    def from_dict(obj: Any) -> 'Span':
        assert isinstance(obj, dict)
        from_character = from_int(obj.get("from_character"))
        to_character = from_int(obj.get("to_character"))
        return Span(from_character, to_character)

    def to_dict(self) -> dict:
        result: dict = {}
        result["from_character"] = from_int(self.from_character)
        result["to_character"] = from_int(self.to_character)
        return result


@dataclass
class EntityPart:
    """(Optional) If the part is an entity reference, reference to the entity.
    If not defined, `text` needs to be defined.
    """
    """(Optional) Name of the entity.
    If not specified, entity_id needs to be specified.
    Ex: `I'd like to visit [New York City](city)` where `city` is the name.
    """
    name: str
    """Text used to reference the entity in the example.
    Ex: `I'd like to visit [New York City](city)` where `New York City` is the text.
    """
    text: str
    """(Optional) Unique identifier of the entity.
    If not specified, the entity key needs to be specified.
    """
    entity_id: Optional[str] = None
    """(Optional) Role of the entity in the example if there are more than one present.
    Ex: `I'm flying from new york city to montreal` could have a `from` and a `to` role.
    """
    role: Optional[str] = None
    """(Optional) Span of the entity in the example.
    If the reference is provided through a `TextPart`, the span is not required
    since parts are presenting the text and entities.
    """
    span: Optional[Span] = None
    """(Optional) Entity value `key_value` that this entity points to. This
    corresponds to the entity value's `key_value` field.
    
    Ex: `I went to [NYC]{"entity": "city", "value": "New York City"}`
    where 'New York City' is the value, 'city' is the key.
    
    If unspecified, the text will be used.
    """
    value: Optional[str] = None
    """(Optional) Entity value unique identifier.
    If not specified, the value will be used if it's specified.
    """
    value_id: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'EntityPart':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        text = from_str(obj.get("text"))
        entity_id = from_union([from_str, from_none], obj.get("entity_id"))
        role = from_union([from_str, from_none], obj.get("role"))
        span = from_union([Span.from_dict, from_none], obj.get("span"))
        value = from_union([from_str, from_none], obj.get("value"))
        value_id = from_union([from_str, from_none], obj.get("value_id"))
        return EntityPart(name, text, entity_id, role, span, value, value_id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["text"] = from_str(self.text)
        if self.entity_id is not None:
            result["entity_id"] = from_union([from_str, from_none], self.entity_id)
        if self.role is not None:
            result["role"] = from_union([from_str, from_none], self.role)
        if self.span is not None:
            result["span"] = from_union([lambda x: to_class(Span, x), from_none], self.span)
        if self.value is not None:
            result["value"] = from_union([from_str, from_none], self.value)
        if self.value_id is not None:
            result["value_id"] = from_union([from_str, from_none], self.value_id)
        return result


@dataclass
class PartElement:
    """(Optional) If the part is an entity reference, reference to the entity.
    If not defined, `text` needs to be defined.
    """
    entity: Optional[EntityPart] = None
    """(Optional) Text of the part if not an entity. If not defined, `entity` needs to be
    defined.
    """
    text: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'PartElement':
        assert isinstance(obj, dict)
        entity = from_union([EntityPart.from_dict, from_none], obj.get("entity"))
        text = from_union([from_str, from_none], obj.get("text"))
        return PartElement(entity, text)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.entity is not None:
            result["entity"] = from_union([lambda x: to_class(EntityPart, x), from_none], self.entity)
        if self.text is not None:
            result["text"] = from_union([from_str, from_none], self.text)
        return result


@dataclass
class ValueSource:
    """(Optional) Information about the source of the entity value."""
    """(Optional) Extra metadata about the entity value in the source system."""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) Unique identifier of the entity in the source system."""
    source_id: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ValueSource':
        assert isinstance(obj, dict)
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        source_id = from_union([from_str, from_none], obj.get("source_id"))
        return ValueSource(metadata, source_id)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.source_id is not None:
            result["source_id"] = from_union([from_str, from_none], self.source_id)
        return result


@dataclass
class SynonymElement:
    """Value / text of the synonym."""
    value: str
    """(Optional) Entities annotated in the `value`. This is used
    when the entity is a composed entity that references other
    entities.
    
    If the `parts` field is provided at import, this field is ignored and
    built from `parts`.
    """
    entities: Optional[List[EntityPart]] = None
    """(Optional) Parts of the text and the entities constituting `value`.
    This is used when the entity is a composed entity that references other
    entities (composite entities).
    
    The parts are concatenated to form the final `value`. Parts are provided
    to ease entity annotations. If provided at import, this will take precedence
    over the `entities` field.
    """
    parts: Optional[List[PartElement]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SynonymElement':
        assert isinstance(obj, dict)
        value = from_str(obj.get("value"))
        entities = from_union([lambda x: from_list(EntityPart.from_dict, x), from_none], obj.get("entities"))
        parts = from_union([lambda x: from_list(PartElement.from_dict, x), from_none], obj.get("parts"))
        return SynonymElement(value, entities, parts)

    def to_dict(self) -> dict:
        result: dict = {}
        result["value"] = from_str(self.value)
        if self.entities is not None:
            result["entities"] = from_union([lambda x: from_list(lambda x: to_class(EntityPart, x), x), from_none], self.entities)
        if self.parts is not None:
            result["parts"] = from_union([lambda x: from_list(lambda x: to_class(PartElement, x), x), from_none], self.parts)
        return result


@dataclass
class ValueElement:
    """(Optional) Creation, update and deletion timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    created_at: Optional[str] = None
    deleted_at: Optional[str] = None
    """(Optional) If supported by the NLU engine, entities referenced in `key_value`.
    Ex: DialogFlow composite entities are referenced here.
    
    If the `key_value_parts` field is provided at import, this field is ignored and
    built from `key_value_parts`.
    """
    entities: Optional[List[EntityPart]] = None
    """(Optional) Unique identifier of the entity value.
    The identifier will be translated into a deterministic internal id
    at import time. Any references to this intent will also be translated.
    Use `source.source_id` to store the original id.
    """
    id: Optional[str] = None
    """(Optional) Key value of the entity. This corresponds to the main disambiguated
    value of the entity value.
    Ex: New York City
    """
    key_value: Optional[str] = None
    """(Optional) If supported by the NLU engine and that `key_value` contains entity
    references,
    this field contains the parts of the text and the entities. The parts are concatenated
    to form the final text.
    
    Parts are provided to ease entity annotations. If provided at import, this will take
    precedence over the `entities` field.
    """
    key_value_parts: Optional[List[PartElement]] = None
    """(Optional) Language of the entity value.
    This is in two-letter ISO 639-1 format.
    """
    language: Optional[str] = None
    """(Optional) Information about the source of the entity value."""
    source: Optional[ValueSource] = None
    """(Optional) Synonyms of the entity value.
    Ex: New York, NYC, The Big Apple
    """
    synonyms: Optional[List[SynonymElement]] = None
    updated_at: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ValueElement':
        assert isinstance(obj, dict)
        created_at = from_union([from_str, from_none], obj.get("created_at"))
        deleted_at = from_union([from_str, from_none], obj.get("deleted_at"))
        entities = from_union([lambda x: from_list(EntityPart.from_dict, x), from_none], obj.get("entities"))
        id = from_union([from_str, from_none], obj.get("id"))
        key_value = from_union([from_str, from_none], obj.get("key_value"))
        key_value_parts = from_union([lambda x: from_list(PartElement.from_dict, x), from_none], obj.get("key_value_parts"))
        language = from_union([from_str, from_none], obj.get("language"))
        source = from_union([ValueSource.from_dict, from_none], obj.get("source"))
        synonyms = from_union([lambda x: from_list(SynonymElement.from_dict, x), from_none], obj.get("synonyms"))
        updated_at = from_union([from_str, from_none], obj.get("updated_at"))
        return ValueElement(created_at, deleted_at, entities, id, key_value, key_value_parts, language, source, synonyms, updated_at)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.created_at is not None:
            result["created_at"] = from_union([from_str, from_none], self.created_at)
        if self.deleted_at is not None:
            result["deleted_at"] = from_union([from_str, from_none], self.deleted_at)
        if self.entities is not None:
            result["entities"] = from_union([lambda x: from_list(lambda x: to_class(EntityPart, x), x), from_none], self.entities)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.key_value is not None:
            result["key_value"] = from_union([from_str, from_none], self.key_value)
        if self.key_value_parts is not None:
            result["key_value_parts"] = from_union([lambda x: from_list(lambda x: to_class(PartElement, x), x), from_none], self.key_value_parts)
        if self.language is not None:
            result["language"] = from_union([from_str, from_none], self.language)
        if self.source is not None:
            result["source"] = from_union([lambda x: to_class(ValueSource, x), from_none], self.source)
        if self.synonyms is not None:
            result["synonyms"] = from_union([lambda x: from_list(lambda x: to_class(SynonymElement, x), x), from_none], self.synonyms)
        if self.updated_at is not None:
            result["updated_at"] = from_union([from_str, from_none], self.updated_at)
        return result


@dataclass
class WorkspaceEntity:
    """Name of the entity.
    Ex: city
    """
    name: str
    """(Optional) Creation timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    created_at: Optional[str] = None
    """(Optional) Deletion timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    deleted_at: Optional[str] = None
    """(Optional) Unique identifier of the entity.
    The identifier will be translated into a deterministic internal id
    at import time. Any references to this entity will also be translated.
    Use `source.source_id` to store the original id.
    """
    id: Optional[str] = None
    """Indicates that the entity of type regex. Values of the entity are regular expressions."""
    is_regex: Optional[bool] = None
    """(Optional) Key-value metadata of the entity."""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) Information about the source of the entity."""
    source: Optional[EntitySource] = None
    """(Optional) If specified, indicates that the is a pre-trained system entity (ex: time,
    date, etc.)
    This field will contain the type of the system entity, as defined in HumanFirst.
    """
    system_type: Optional[str] = None
    """(Optional) List of tags for the entity."""
    tags: Optional[List[EntityTag]] = None
    """(Optional) Update timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    updated_at: Optional[str] = None
    """Values of the entity. They represent the different instances of the entity.
    Ex: new york, montreal, etc.
    
    For regex entities, each value represent a different regular expression.
    
    For system entities, values can be used to extend the potential values of the
    entity, but only if the extension is possible for the type of system entity.
    """
    values: Optional[List[ValueElement]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'WorkspaceEntity':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        created_at = from_union([from_str, from_none], obj.get("created_at"))
        deleted_at = from_union([from_str, from_none], obj.get("deleted_at"))
        id = from_union([from_str, from_none], obj.get("id"))
        is_regex = from_union([from_bool, from_none], obj.get("is_regex"))
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        source = from_union([EntitySource.from_dict, from_none], obj.get("source"))
        system_type = from_union([from_str, from_none], obj.get("system_type"))
        tags = from_union([lambda x: from_list(EntityTag.from_dict, x), from_none], obj.get("tags"))
        updated_at = from_union([from_str, from_none], obj.get("updated_at"))
        values = from_union([lambda x: from_list(ValueElement.from_dict, x), from_none], obj.get("values"))
        return WorkspaceEntity(name, created_at, deleted_at, id, is_regex, metadata, source, system_type, tags, updated_at, values)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        if self.created_at is not None:
            result["created_at"] = from_union([from_str, from_none], self.created_at)
        if self.deleted_at is not None:
            result["deleted_at"] = from_union([from_str, from_none], self.deleted_at)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.is_regex is not None:
            result["is_regex"] = from_union([from_bool, from_none], self.is_regex)
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.source is not None:
            result["source"] = from_union([lambda x: to_class(EntitySource, x), from_none], self.source)
        if self.system_type is not None:
            result["system_type"] = from_union([from_str, from_none], self.system_type)
        if self.tags is not None:
            result["tags"] = from_union([lambda x: from_list(lambda x: to_class(EntityTag, x), x), from_none], self.tags)
        if self.updated_at is not None:
            result["updated_at"] = from_union([from_str, from_none], self.updated_at)
        if self.values is not None:
            result["values"] = from_union([lambda x: from_list(lambda x: to_class(ValueElement, x), x), from_none], self.values)
        return result


@dataclass
class Context:
    """(Optional) Information on the context of the example."""
    """(Optional) Unique identifier of the context in which the example is used.
    Ex: conversation id in which the example was seen.
    """
    context_id: Optional[str] = None
    """(Optional) In the case of an utterance / conversation input, role of the person who did
    the
    utterance. Ex: "client" "expert"
    """
    role: Optional[str] = None
    """(Optional) Type of context in which the example is used.
    Ex: conversation, utterance, training_phrase
    """
    type: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Context':
        assert isinstance(obj, dict)
        context_id = from_union([from_str, from_none], obj.get("context_id"))
        role = from_union([from_str, from_none], obj.get("role"))
        type = from_union([from_str, from_none], obj.get("type"))
        return Context(context_id, role, type)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.context_id is not None:
            result["context_id"] = from_union([from_str, from_none], self.context_id)
        if self.role is not None:
            result["role"] = from_union([from_str, from_none], self.role)
        if self.type is not None:
            result["type"] = from_union([from_str, from_none], self.type)
        return result


@dataclass
class ExampleIntent:
    """Unique identifier of the intent."""
    intent_id: str
    """Indicates if the example is in the negative examples of the intent.
    A negative example is only used to train the intent and is created by rejecting
    suggestions in HumanFirst studio.
    """
    negative: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ExampleIntent':
        assert isinstance(obj, dict)
        intent_id = from_str(obj.get("intent_id"))
        negative = from_union([from_bool, from_none], obj.get("negative"))
        return ExampleIntent(intent_id, negative)

    def to_dict(self) -> dict:
        result: dict = {}
        result["intent_id"] = from_str(self.intent_id)
        if self.negative is not None:
            result["negative"] = from_union([from_bool, from_none], self.negative)
        return result


@dataclass
class ExampleSource:
    """(Optional) Information on the source of the example."""
    """(Optional) Unique identifier of the example in the source system."""
    source_id: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ExampleSource':
        assert isinstance(obj, dict)
        source_id = from_union([from_str, from_none], obj.get("source_id"))
        return ExampleSource(source_id)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.source_id is not None:
            result["source_id"] = from_union([from_str, from_none], self.source_id)
        return result


@dataclass
class ExampleElement:
    """Text of the example."""
    text: str
    """(Optional) Information on the context of the example."""
    context: Optional[Context] = None
    """(Optional) Creation timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    created_at: Optional[str] = None
    """(Optional) Deletion timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    deleted_at: Optional[str] = None
    """(Optional) Entities annotated in the example.
    If the `parts` field is provided at import, this field is ignored and built from `parts`.
    """
    entities: Optional[List[EntityPart]] = None
    """(Optional) Unique identifier of the example.
    The identifier will be translated into a deterministic internal id
    at import time. Any references to this example will also be translated.
    Use `source.source_id` to store the original id.
    """
    id: Optional[str] = None
    """(Optional) Intents into which the example is classified."""
    intents: Optional[List[ExampleIntent]] = None
    """(Optional) Key-value metadata of the example."""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) If the example contains entities, this field contains parts of the
    text and the entities. The parts are concatenated to form the final text.
    
    Parts are provided to ease entity annotations. If provided at import, this
    will take precedence over the `entities` field.
    """
    parts: Optional[List[PartElement]] = None
    """(Optional) Information on the source of the example."""
    source: Optional[ExampleSource] = None
    """(Optional) List of tags for the example."""
    tags: Optional[List[EntityTag]] = None
    """(Optional) Update timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    updated_at: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ExampleElement':
        assert isinstance(obj, dict)
        text = from_str(obj.get("text"))
        context = from_union([Context.from_dict, from_none], obj.get("context"))
        created_at = from_union([from_str, from_none], obj.get("created_at"))
        deleted_at = from_union([from_str, from_none], obj.get("deleted_at"))
        entities = from_union([lambda x: from_list(EntityPart.from_dict, x), from_none], obj.get("entities"))
        id = from_union([from_str, from_none], obj.get("id"))
        intents = from_union([lambda x: from_list(ExampleIntent.from_dict, x), from_none], obj.get("intents"))
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        parts = from_union([lambda x: from_list(PartElement.from_dict, x), from_none], obj.get("parts"))
        source = from_union([ExampleSource.from_dict, from_none], obj.get("source"))
        tags = from_union([lambda x: from_list(EntityTag.from_dict, x), from_none], obj.get("tags"))
        updated_at = from_union([from_str, from_none], obj.get("updated_at"))
        return ExampleElement(text, context, created_at, deleted_at, entities, id, intents, metadata, parts, source, tags, updated_at)

    def to_dict(self) -> dict:
        result: dict = {}
        result["text"] = from_str(self.text)
        if self.context is not None:
            result["context"] = from_union([lambda x: to_class(Context, x), from_none], self.context)
        if self.created_at is not None:
            result["created_at"] = from_union([from_str, from_none], self.created_at)
        if self.deleted_at is not None:
            result["deleted_at"] = from_union([from_str, from_none], self.deleted_at)
        if self.entities is not None:
            result["entities"] = from_union([lambda x: from_list(lambda x: to_class(EntityPart, x), x), from_none], self.entities)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.intents is not None:
            result["intents"] = from_union([lambda x: from_list(lambda x: to_class(ExampleIntent, x), x), from_none], self.intents)
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.parts is not None:
            result["parts"] = from_union([lambda x: from_list(lambda x: to_class(PartElement, x), x), from_none], self.parts)
        if self.source is not None:
            result["source"] = from_union([lambda x: to_class(ExampleSource, x), from_none], self.source)
        if self.tags is not None:
            result["tags"] = from_union([lambda x: from_list(lambda x: to_class(EntityTag, x), x), from_none], self.tags)
        if self.updated_at is not None:
            result["updated_at"] = from_union([from_str, from_none], self.updated_at)
        return result


@dataclass
class IntentSource:
    """(Optional) Information on the source of the intent."""
    """(Optional) Extra metadata about the intent in the source system."""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) Unique identifier of the intent in the source system."""
    source_id: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'IntentSource':
        assert isinstance(obj, dict)
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        source_id = from_union([from_str, from_none], obj.get("source_id"))
        return IntentSource(metadata, source_id)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.source_id is not None:
            result["source_id"] = from_union([from_str, from_none], self.source_id)
        return result


@dataclass
class WorkspaceIntent:
    """Name of the intent."""
    name: str
    """(Optional) Creation timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    created_at: Optional[str] = None
    """(Optional) Deletion timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    deleted_at: Optional[str] = None
    """(Optional) Description of the intent."""
    description: Optional[str] = None
    """(Optional) Unique identifier of the intent.
    The identifier will be translated into a deterministic internal id
    at import time. Any references to this intent will also be translated.
    Use `source.source_id` to store the original id.
    """
    id: Optional[str] = None
    """(Optional) Key-value metadata of the intent."""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) Unique identifier of the intent in the hierarchy."""
    parent_intent_id: Optional[str] = None
    """(Optional) Information on the source of the intent."""
    source: Optional[IntentSource] = None
    """(Optional) List of tags for the intent."""
    tags: Optional[List[EntityTag]] = None
    """(Optional) Update timestamps.
    The timestamps are in RFC3339 format ("2006-01-02T15:04:05Z07:00")
    """
    updated_at: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'WorkspaceIntent':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        created_at = from_union([from_str, from_none], obj.get("created_at"))
        deleted_at = from_union([from_str, from_none], obj.get("deleted_at"))
        description = from_union([from_str, from_none], obj.get("description"))
        id = from_union([from_str, from_none], obj.get("id"))
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        parent_intent_id = from_union([from_str, from_none], obj.get("parent_intent_id"))
        source = from_union([IntentSource.from_dict, from_none], obj.get("source"))
        tags = from_union([lambda x: from_list(EntityTag.from_dict, x), from_none], obj.get("tags"))
        updated_at = from_union([from_str, from_none], obj.get("updated_at"))
        return WorkspaceIntent(name, created_at, deleted_at, description, id, metadata, parent_intent_id, source, tags, updated_at)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        if self.created_at is not None:
            result["created_at"] = from_union([from_str, from_none], self.created_at)
        if self.deleted_at is not None:
            result["deleted_at"] = from_union([from_str, from_none], self.deleted_at)
        if self.description is not None:
            result["description"] = from_union([from_str, from_none], self.description)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.parent_intent_id is not None:
            result["parent_intent_id"] = from_union([from_str, from_none], self.parent_intent_id)
        if self.source is not None:
            result["source"] = from_union([lambda x: to_class(IntentSource, x), from_none], self.source)
        if self.tags is not None:
            result["tags"] = from_union([lambda x: from_list(lambda x: to_class(EntityTag, x), x), from_none], self.tags)
        if self.updated_at is not None:
            result["updated_at"] = from_union([from_str, from_none], self.updated_at)
        return result


@dataclass
class WorkspaceTag:
    """Name of the tag."""
    name: str
    """(Optional) Color of the tag, that can be of any valid css color format.
    Ex: red, blue, #133337, etc.
    """
    color: Optional[str] = None
    """(Optional) Description of the tag."""
    description: Optional[str] = None
    """(Optional) Unique identifier of the tag.
    The identifier will be translated into a deterministic internal id
    at import time. Any references to this tag will also be translated.
    """
    id: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'WorkspaceTag':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        color = from_union([from_str, from_none], obj.get("color"))
        description = from_union([from_str, from_none], obj.get("description"))
        id = from_union([from_str, from_none], obj.get("id"))
        return WorkspaceTag(name, color, description, id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        if self.color is not None:
            result["color"] = from_union([from_str, from_none], self.color)
        if self.description is not None:
            result["description"] = from_union([from_str, from_none], self.description)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        return result


@dataclass
class Workspace:
    """(Optional) Color of the workspace"""
    color: Optional[str] = None
    """(Optional) Description of the workspace"""
    description: Optional[str] = None
    """(Optional) Entities of the workspace"""
    entities: Optional[List[WorkspaceEntity]] = None
    """(Optional) Examples (labelled or unlabelled) of the workspace"""
    examples: Optional[List[ExampleElement]] = None
    """(Optional) Intents of the workspace"""
    intents: Optional[List[WorkspaceIntent]] = None
    """(Optional) Metadata of the workspace"""
    metadata: Optional[Dict[str, str]] = None
    """(Optional) Name of the workspace"""
    name: Optional[str] = None
    """(Optional) Tags of the workspace"""
    tags: Optional[List[WorkspaceTag]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Workspace':
        assert isinstance(obj, dict)
        color = from_union([from_str, from_none], obj.get("color"))
        description = from_union([from_str, from_none], obj.get("description"))
        entities = from_union([lambda x: from_list(WorkspaceEntity.from_dict, x), from_none], obj.get("entities"))
        examples = from_union([lambda x: from_list(ExampleElement.from_dict, x), from_none], obj.get("examples"))
        intents = from_union([lambda x: from_list(WorkspaceIntent.from_dict, x), from_none], obj.get("intents"))
        metadata = from_union([lambda x: from_dict(from_str, x), from_none], obj.get("metadata"))
        name = from_union([from_str, from_none], obj.get("name"))
        tags = from_union([lambda x: from_list(WorkspaceTag.from_dict, x), from_none], obj.get("tags"))
        return Workspace(color, description, entities, examples, intents, metadata, name, tags)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.color is not None:
            result["color"] = from_union([from_str, from_none], self.color)
        if self.description is not None:
            result["description"] = from_union([from_str, from_none], self.description)
        if self.entities is not None:
            result["entities"] = from_union([lambda x: from_list(lambda x: to_class(WorkspaceEntity, x), x), from_none], self.entities)
        if self.examples is not None:
            result["examples"] = from_union([lambda x: from_list(lambda x: to_class(ExampleElement, x), x), from_none], self.examples)
        if self.intents is not None:
            result["intents"] = from_union([lambda x: from_list(lambda x: to_class(WorkspaceIntent, x), x), from_none], self.intents)
        if self.metadata is not None:
            result["metadata"] = from_union([lambda x: from_dict(from_str, x), from_none], self.metadata)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.tags is not None:
            result["tags"] = from_union([lambda x: from_list(lambda x: to_class(WorkspaceTag, x), x), from_none], self.tags)
        return result


def workspace_from_dict(s: Any) -> Workspace:
    return Workspace.from_dict(s)


def workspace_to_dict(x: Workspace) -> Any:
    return to_class(Workspace, x)
