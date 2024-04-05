# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tags/tags.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import field_mask_pb2 as google_dot_protobuf_dot_field__mask__pb2
from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0ftags/tags.proto\x12\x0bzia.ai.tags\x1a\x1fgoogle/protobuf/timestamp.proto\x1a google/protobuf/field_mask.proto\x1a\x1egoogle/protobuf/wrappers.proto\x1a\x1cgoogle/api/annotations.proto\"\xab\x02\n\x03Tag\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x03 \x01(\t\x12\r\n\x05\x63olor\x18\x04 \x01(\t\x12\x11\n\tprotected\x18\x08 \x01(\x08\x12\x1b\n\x13protected_recursive\x18\t \x01(\x08\x12&\n\x06source\x18\n \x01(\x0b\x32\x16.zia.ai.tags.TagSource\x12.\n\ncreated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12.\n\nupdated_at\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12.\n\ndeleted_at\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"\x95\x05\n\tTagSource\x12\x11\n\tsource_id\x18\x01 \x01(\t\x12\x12\n\nmerged_ids\x18\x02 \x03(\t\x12<\n\rdialogflow_cx\x18\x03 \x01(\x0b\x32#.zia.ai.tags.TagSource.DialogflowCxH\x00\x12<\n\rdialogflow_es\x18\x04 \x01(\x0b\x32#.zia.ai.tags.TagSource.DialogflowEsH\x00\x1aS\n\x0c\x44ialogflowCx\x12\x0f\n\x07\x66low_id\x18\x01 \x01(\t\x12\x0f\n\x07page_id\x18\x02 \x01(\t\x12!\n\x19transition_route_group_id\x18\x03 \x01(\t\x1a\x87\x03\n\x0c\x44ialogflowEs\x12@\n\x08priority\x18\x01 \x01(\x0b\x32,.zia.ai.tags.TagSource.DialogflowEs.PriorityH\x00\x12\x41\n\ttop_level\x18\x02 \x01(\x0b\x32,.zia.ai.tags.TagSource.DialogflowEs.TopLevelH\x00\x12>\n\x07\x63ontext\x18\x03 \x01(\x0b\x32+.zia.ai.tags.TagSource.DialogflowEs.ContextH\x00\x12\x41\n\tfollow_up\x18\x04 \x01(\x0b\x32,.zia.ai.tags.TagSource.DialogflowEs.FollowUpH\x00\x1a\x36\n\x08Priority\x12*\n\x05value\x18\x01 \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x1a\n\n\x08TopLevel\x1a\x17\n\x07\x43ontext\x12\x0c\n\x04name\x18\x01 \x01(\t\x1a\n\n\x08\x46ollowUpB\x06\n\x04typeB\x06\n\x04type\";\n\x0cTagReference\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x11\n\tprotected\x18\x03 \x01(\x08\"8\n\rTagReferences\x12\'\n\x04tags\x18\x01 \x03(\x0b\x32\x19.zia.ai.tags.TagReference\"M\n\x0cTagPredicate\x12\x13\n\x0brequire_ids\x18\x01 \x03(\t\x12\x13\n\x0binclude_ids\x18\x03 \x03(\t\x12\x13\n\x0b\x65xclude_ids\x18\x02 \x03(\t\"Y\n\x10\x43reateTagRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x13\n\x0bplaybook_id\x18\x02 \x01(\t\x12\x1d\n\x03tag\x18\x03 \x01(\x0b\x32\x10.zia.ai.tags.Tag\"2\n\x11\x43reateTagResponse\x12\x1d\n\x03tag\x18\x01 \x01(\x0b\x32\x10.zia.ai.tags.Tag\"9\n\x0fListTagsRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x13\n\x0bplaybook_id\x18\x02 \x01(\t\"2\n\x10ListTagsResponse\x12\x1e\n\x04tags\x18\x01 \x03(\x0b\x32\x10.zia.ai.tags.Tag\"\x9e\x01\n\x10UpdateTagRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x13\n\x0bplaybook_id\x18\x02 \x01(\t\x12\x1d\n\x03tag\x18\x03 \x01(\x0b\x32\x10.zia.ai.tags.Tag\x12/\n\x0bupdate_mask\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.FieldMask\x12\x12\n\nupdate_all\x18\x05 \x01(\x08\"2\n\x11UpdateTagResponse\x12\x1d\n\x03tag\x18\x01 \x01(\x0b\x32\x10.zia.ai.tags.Tag\"J\n\x10\x44\x65leteTagRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x13\n\x0bplaybook_id\x18\x02 \x01(\t\x12\x0e\n\x06tag_id\x18\x03 \x01(\t\"\x13\n\x11\x44\x65leteTagResponse2\xbd\x04\n\x04Tags\x12\x87\x01\n\tCreateTag\x12\x1d.zia.ai.tags.CreateTagRequest\x1a\x1e.zia.ai.tags.CreateTagResponse\";\x82\xd3\xe4\x93\x02\x35\"3/v1alpha1/workspaces/{namespace}/{playbook_id}/tags\x12\x84\x01\n\x08ListTags\x12\x1c.zia.ai.tags.ListTagsRequest\x1a\x1d.zia.ai.tags.ListTagsResponse\";\x82\xd3\xe4\x93\x02\x35\x12\x33/v1alpha1/workspaces/{namespace}/{playbook_id}/tags\x12\x90\x01\n\tUpdateTag\x12\x1d.zia.ai.tags.UpdateTagRequest\x1a\x1e.zia.ai.tags.UpdateTagResponse\"D\x82\xd3\xe4\x93\x02>\x1a</v1alpha1/workspaces/{namespace}/{playbook_id}/tags/{tag.id}\x12\x90\x01\n\tDeleteTag\x12\x1d.zia.ai.tags.DeleteTagRequest\x1a\x1e.zia.ai.tags.DeleteTagResponse\"D\x82\xd3\xe4\x93\x02>*</v1alpha1/workspaces/{namespace}/{playbook_id}/tags/{tag_id}B.Z,github.com/zia-ai/platform/pkg/api/tags;tagsb\x06proto3')



_TAG = DESCRIPTOR.message_types_by_name['Tag']
_TAGSOURCE = DESCRIPTOR.message_types_by_name['TagSource']
_TAGSOURCE_DIALOGFLOWCX = _TAGSOURCE.nested_types_by_name['DialogflowCx']
_TAGSOURCE_DIALOGFLOWES = _TAGSOURCE.nested_types_by_name['DialogflowEs']
_TAGSOURCE_DIALOGFLOWES_PRIORITY = _TAGSOURCE_DIALOGFLOWES.nested_types_by_name['Priority']
_TAGSOURCE_DIALOGFLOWES_TOPLEVEL = _TAGSOURCE_DIALOGFLOWES.nested_types_by_name['TopLevel']
_TAGSOURCE_DIALOGFLOWES_CONTEXT = _TAGSOURCE_DIALOGFLOWES.nested_types_by_name['Context']
_TAGSOURCE_DIALOGFLOWES_FOLLOWUP = _TAGSOURCE_DIALOGFLOWES.nested_types_by_name['FollowUp']
_TAGREFERENCE = DESCRIPTOR.message_types_by_name['TagReference']
_TAGREFERENCES = DESCRIPTOR.message_types_by_name['TagReferences']
_TAGPREDICATE = DESCRIPTOR.message_types_by_name['TagPredicate']
_CREATETAGREQUEST = DESCRIPTOR.message_types_by_name['CreateTagRequest']
_CREATETAGRESPONSE = DESCRIPTOR.message_types_by_name['CreateTagResponse']
_LISTTAGSREQUEST = DESCRIPTOR.message_types_by_name['ListTagsRequest']
_LISTTAGSRESPONSE = DESCRIPTOR.message_types_by_name['ListTagsResponse']
_UPDATETAGREQUEST = DESCRIPTOR.message_types_by_name['UpdateTagRequest']
_UPDATETAGRESPONSE = DESCRIPTOR.message_types_by_name['UpdateTagResponse']
_DELETETAGREQUEST = DESCRIPTOR.message_types_by_name['DeleteTagRequest']
_DELETETAGRESPONSE = DESCRIPTOR.message_types_by_name['DeleteTagResponse']
Tag = _reflection.GeneratedProtocolMessageType('Tag', (_message.Message,), {
  'DESCRIPTOR' : _TAG,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.Tag)
  })
_sym_db.RegisterMessage(Tag)

TagSource = _reflection.GeneratedProtocolMessageType('TagSource', (_message.Message,), {

  'DialogflowCx' : _reflection.GeneratedProtocolMessageType('DialogflowCx', (_message.Message,), {
    'DESCRIPTOR' : _TAGSOURCE_DIALOGFLOWCX,
    '__module__' : 'tags.tags_pb2'
    # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource.DialogflowCx)
    })
  ,

  'DialogflowEs' : _reflection.GeneratedProtocolMessageType('DialogflowEs', (_message.Message,), {

    'Priority' : _reflection.GeneratedProtocolMessageType('Priority', (_message.Message,), {
      'DESCRIPTOR' : _TAGSOURCE_DIALOGFLOWES_PRIORITY,
      '__module__' : 'tags.tags_pb2'
      # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource.DialogflowEs.Priority)
      })
    ,

    'TopLevel' : _reflection.GeneratedProtocolMessageType('TopLevel', (_message.Message,), {
      'DESCRIPTOR' : _TAGSOURCE_DIALOGFLOWES_TOPLEVEL,
      '__module__' : 'tags.tags_pb2'
      # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource.DialogflowEs.TopLevel)
      })
    ,

    'Context' : _reflection.GeneratedProtocolMessageType('Context', (_message.Message,), {
      'DESCRIPTOR' : _TAGSOURCE_DIALOGFLOWES_CONTEXT,
      '__module__' : 'tags.tags_pb2'
      # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource.DialogflowEs.Context)
      })
    ,

    'FollowUp' : _reflection.GeneratedProtocolMessageType('FollowUp', (_message.Message,), {
      'DESCRIPTOR' : _TAGSOURCE_DIALOGFLOWES_FOLLOWUP,
      '__module__' : 'tags.tags_pb2'
      # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource.DialogflowEs.FollowUp)
      })
    ,
    'DESCRIPTOR' : _TAGSOURCE_DIALOGFLOWES,
    '__module__' : 'tags.tags_pb2'
    # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource.DialogflowEs)
    })
  ,
  'DESCRIPTOR' : _TAGSOURCE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.TagSource)
  })
_sym_db.RegisterMessage(TagSource)
_sym_db.RegisterMessage(TagSource.DialogflowCx)
_sym_db.RegisterMessage(TagSource.DialogflowEs)
_sym_db.RegisterMessage(TagSource.DialogflowEs.Priority)
_sym_db.RegisterMessage(TagSource.DialogflowEs.TopLevel)
_sym_db.RegisterMessage(TagSource.DialogflowEs.Context)
_sym_db.RegisterMessage(TagSource.DialogflowEs.FollowUp)

TagReference = _reflection.GeneratedProtocolMessageType('TagReference', (_message.Message,), {
  'DESCRIPTOR' : _TAGREFERENCE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.TagReference)
  })
_sym_db.RegisterMessage(TagReference)

TagReferences = _reflection.GeneratedProtocolMessageType('TagReferences', (_message.Message,), {
  'DESCRIPTOR' : _TAGREFERENCES,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.TagReferences)
  })
_sym_db.RegisterMessage(TagReferences)

TagPredicate = _reflection.GeneratedProtocolMessageType('TagPredicate', (_message.Message,), {
  'DESCRIPTOR' : _TAGPREDICATE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.TagPredicate)
  })
_sym_db.RegisterMessage(TagPredicate)

CreateTagRequest = _reflection.GeneratedProtocolMessageType('CreateTagRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATETAGREQUEST,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.CreateTagRequest)
  })
_sym_db.RegisterMessage(CreateTagRequest)

CreateTagResponse = _reflection.GeneratedProtocolMessageType('CreateTagResponse', (_message.Message,), {
  'DESCRIPTOR' : _CREATETAGRESPONSE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.CreateTagResponse)
  })
_sym_db.RegisterMessage(CreateTagResponse)

ListTagsRequest = _reflection.GeneratedProtocolMessageType('ListTagsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTTAGSREQUEST,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.ListTagsRequest)
  })
_sym_db.RegisterMessage(ListTagsRequest)

ListTagsResponse = _reflection.GeneratedProtocolMessageType('ListTagsResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTTAGSRESPONSE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.ListTagsResponse)
  })
_sym_db.RegisterMessage(ListTagsResponse)

UpdateTagRequest = _reflection.GeneratedProtocolMessageType('UpdateTagRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATETAGREQUEST,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.UpdateTagRequest)
  })
_sym_db.RegisterMessage(UpdateTagRequest)

UpdateTagResponse = _reflection.GeneratedProtocolMessageType('UpdateTagResponse', (_message.Message,), {
  'DESCRIPTOR' : _UPDATETAGRESPONSE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.UpdateTagResponse)
  })
_sym_db.RegisterMessage(UpdateTagResponse)

DeleteTagRequest = _reflection.GeneratedProtocolMessageType('DeleteTagRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETETAGREQUEST,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.DeleteTagRequest)
  })
_sym_db.RegisterMessage(DeleteTagRequest)

DeleteTagResponse = _reflection.GeneratedProtocolMessageType('DeleteTagResponse', (_message.Message,), {
  'DESCRIPTOR' : _DELETETAGRESPONSE,
  '__module__' : 'tags.tags_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.tags.DeleteTagResponse)
  })
_sym_db.RegisterMessage(DeleteTagResponse)

_TAGS = DESCRIPTOR.services_by_name['Tags']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z,github.com/zia-ai/platform/pkg/api/tags;tags'
  _TAGS.methods_by_name['CreateTag']._options = None
  _TAGS.methods_by_name['CreateTag']._serialized_options = b'\202\323\344\223\0025\"3/v1alpha1/workspaces/{namespace}/{playbook_id}/tags'
  _TAGS.methods_by_name['ListTags']._options = None
  _TAGS.methods_by_name['ListTags']._serialized_options = b'\202\323\344\223\0025\0223/v1alpha1/workspaces/{namespace}/{playbook_id}/tags'
  _TAGS.methods_by_name['UpdateTag']._options = None
  _TAGS.methods_by_name['UpdateTag']._serialized_options = b'\202\323\344\223\002>\032</v1alpha1/workspaces/{namespace}/{playbook_id}/tags/{tag.id}'
  _TAGS.methods_by_name['DeleteTag']._options = None
  _TAGS.methods_by_name['DeleteTag']._serialized_options = b'\202\323\344\223\002>*</v1alpha1/workspaces/{namespace}/{playbook_id}/tags/{tag_id}'
  _TAG._serialized_start=162
  _TAG._serialized_end=461
  _TAGSOURCE._serialized_start=464
  _TAGSOURCE._serialized_end=1125
  _TAGSOURCE_DIALOGFLOWCX._serialized_start=640
  _TAGSOURCE_DIALOGFLOWCX._serialized_end=723
  _TAGSOURCE_DIALOGFLOWES._serialized_start=726
  _TAGSOURCE_DIALOGFLOWES._serialized_end=1117
  _TAGSOURCE_DIALOGFLOWES_PRIORITY._serialized_start=1006
  _TAGSOURCE_DIALOGFLOWES_PRIORITY._serialized_end=1060
  _TAGSOURCE_DIALOGFLOWES_TOPLEVEL._serialized_start=1062
  _TAGSOURCE_DIALOGFLOWES_TOPLEVEL._serialized_end=1072
  _TAGSOURCE_DIALOGFLOWES_CONTEXT._serialized_start=1074
  _TAGSOURCE_DIALOGFLOWES_CONTEXT._serialized_end=1097
  _TAGSOURCE_DIALOGFLOWES_FOLLOWUP._serialized_start=1099
  _TAGSOURCE_DIALOGFLOWES_FOLLOWUP._serialized_end=1109
  _TAGREFERENCE._serialized_start=1127
  _TAGREFERENCE._serialized_end=1186
  _TAGREFERENCES._serialized_start=1188
  _TAGREFERENCES._serialized_end=1244
  _TAGPREDICATE._serialized_start=1246
  _TAGPREDICATE._serialized_end=1323
  _CREATETAGREQUEST._serialized_start=1325
  _CREATETAGREQUEST._serialized_end=1414
  _CREATETAGRESPONSE._serialized_start=1416
  _CREATETAGRESPONSE._serialized_end=1466
  _LISTTAGSREQUEST._serialized_start=1468
  _LISTTAGSREQUEST._serialized_end=1525
  _LISTTAGSRESPONSE._serialized_start=1527
  _LISTTAGSRESPONSE._serialized_end=1577
  _UPDATETAGREQUEST._serialized_start=1580
  _UPDATETAGREQUEST._serialized_end=1738
  _UPDATETAGRESPONSE._serialized_start=1740
  _UPDATETAGRESPONSE._serialized_end=1790
  _DELETETAGREQUEST._serialized_start=1792
  _DELETETAGREQUEST._serialized_end=1866
  _DELETETAGRESPONSE._serialized_start=1868
  _DELETETAGRESPONSE._serialized_end=1887
  _TAGS._serialized_start=1890
  _TAGS._serialized_end=2463
# @@protoc_insertion_point(module_scope)
