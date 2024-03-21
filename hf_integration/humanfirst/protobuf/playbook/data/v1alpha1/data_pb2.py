# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: playbook/data/v1alpha1/data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2
from google.api import httpbody_pb2 as google_dot_api_dot_httpbody__pb2
from model import core_pb2 as model_dot_core__pb2
from longrunning.v1alpha1 import operations_pb2 as longrunning_dot_v1alpha1_dot_operations__pb2
from tags import tags_pb2 as tags_dot_tags__pb2
from playbook import training_phrase_pb2 as playbook_dot_training__phrase__pb2
from playbook.data.config.v1alpha1 import config_pb2 as playbook_dot_data_dot_config_dot_v1alpha1_dot_config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n!playbook/data/v1alpha1/data.proto\x12\x1dzia.ai.playbook.data.v1alpha1\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cgoogle/api/annotations.proto\x1a\x19google/api/httpbody.proto\x1a\x10model/core.proto\x1a%longrunning/v1alpha1/operations.proto\x1a\x0ftags/tags.proto\x1a\x1eplaybook/training_phrase.proto\x1a*playbook/data/config/v1alpha1/config.proto\"\xed\x01\n\x14\x45xportIntentsRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x13\n\x0bplaybook_id\x18\x02 \x01(\t\x12G\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x37.zia.ai.playbook.data.config.v1alpha1.IntentsDataFormat\x12P\n\x0e\x66ormat_options\x18\x05 \x01(\x0b\x32\x38.zia.ai.playbook.data.config.v1alpha1.IntentsDataOptions\x12\x12\n\nintent_ids\x18\x04 \x03(\t\"\x81\x01\n\x15\x45xportIntentsResponse\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x42\n\x08problems\x18\x02 \x03(\x0b\x32\x30.zia.ai.playbook.data.v1alpha1.ValidationProblem\x12\x16\n\x0etotal_problems\x18\x03 \x01(\r\"\xf6\x04\n\x14ImportIntentsRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x13\n\x0bplaybook_id\x18\x02 \x01(\t\x12G\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x37.zia.ai.playbook.data.config.v1alpha1.IntentsDataFormat\x12P\n\x0e\x66ormat_options\x18\x04 \x01(\x0b\x32\x38.zia.ai.playbook.data.config.v1alpha1.IntentsDataOptions\x12K\n\x0eimport_options\x18\x10 \x01(\x0b\x32\x33.zia.ai.playbook.data.config.v1alpha1.ImportOptions\x12\x0c\n\x04\x64\x61ta\x18\x05 \x01(\x0c\x12\x1b\n\x0f\x63lear_workspace\x18\x06 \x01(\x08\x42\x02\x18\x01\x12\x19\n\rclear_intents\x18\x0c \x01(\x08\x42\x02\x18\x01\x12\x1a\n\x0e\x63lear_entities\x18\r \x01(\x08\x42\x02\x18\x01\x12\x16\n\nclear_tags\x18\x0e \x01(\x08\x42\x02\x18\x01\x12\x19\n\rmerge_intents\x18\x08 \x01(\x08\x42\x02\x18\x01\x12\x1a\n\x0emerge_entities\x18\t \x01(\x08\x42\x02\x18\x01\x12\x16\n\nmerge_tags\x18\x0f \x01(\x08\x42\x02\x18\x01\x12\x11\n\tsoft_fail\x18\x07 \x01(\x08\x12\x38\n\x11\x65xtra_intent_tags\x18\n \x03(\x0b\x32\x19.zia.ai.tags.TagReferenceB\x02\x18\x01\x12\x38\n\x11\x65xtra_phrase_tags\x18\x0b \x03(\x0b\x32\x19.zia.ai.tags.TagReferenceB\x02\x18\x01\"\x80\x02\n\x15ImportIntentsResponse\x12\x1d\n\x15imported_intent_count\x18\x01 \x01(\r\x12&\n\x1eimported_training_phrase_count\x18\x02 \x01(\r\x12\x42\n\x08problems\x18\x03 \x03(\x0b\x32\x30.zia.ai.playbook.data.v1alpha1.ValidationProblem\x12\x16\n\x0etotal_problems\x18\x04 \x01(\r\x12\x44\n\x14\x62\x61\x63kground_operation\x18\x05 \x01(\x0b\x32&.zia.ai.longrunning.v1alpha1.Operation\"\xbf\x01\n\x1eImportConversationsFileRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x1e\n\x16\x63onversation_source_id\x18\x02 \x01(\t\x12\x10\n\x08\x66ilename\x18\x03 \x01(\t\x12\x37\n\x06\x66ormat\x18\x04 \x01(\x0e\x32\'.zia.ai.model.ConversationsImportFormat\x12\x0c\n\x04\x64\x61ta\x18\x05 \x01(\x0c\x12\x11\n\tsoft_fail\x18\x06 \x01(\x08\"\x8f\x01\n\x1fImportConversationsFileResponse\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x42\n\x08problems\x18\x02 \x03(\x0b\x32\x30.zia.ai.playbook.data.v1alpha1.ValidationProblem\x12\x16\n\x0etotal_problems\x18\x03 \x01(\r\"Q\n\x1cListConversationsFileRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x1e\n\x16\x63onversation_source_id\x18\x02 \x01(\t\"\x8c\x02\n\x1dListConversationsFileResponse\x12P\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x41.zia.ai.playbook.data.v1alpha1.ListConversationsFileResponse.File\x1a\x98\x01\n\x04\x46ile\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x37\n\x06\x66ormat\x18\x02 \x01(\x0e\x32\'.zia.ai.model.ConversationsImportFormat\x12/\n\x0bupload_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x18\n\x10\x66rom_last_upload\x18\x04 \x01(\x08\"e\n\x1e\x44\x65leteConversationsFileRequest\x12\x11\n\tnamespace\x18\x01 \x01(\t\x12\x1e\n\x16\x63onversation_source_id\x18\x02 \x01(\t\x12\x10\n\x08\x66ilename\x18\x03 \x01(\t\"!\n\x1f\x44\x65leteConversationsFileResponse\"\xff\x01\n\x11ValidationProblem\x12\x45\n\x05level\x18\x01 \x01(\x0e\x32\x36.zia.ai.playbook.data.v1alpha1.ValidationProblem.Level\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x10\n\x08\x66ilename\x18\x03 \x01(\t\x12\x0c\n\x04line\x18\x04 \x01(\r\x12:\n\x0ftraining_phrase\x18\x05 \x01(\x0b\x32\x1f.zia.ai.playbook.TrainingPhraseH\x00\",\n\x05Level\x12\x0b\n\x07INVALID\x10\x00\x12\x0b\n\x07WARNING\x10\x01\x12\t\n\x05\x46\x41TAL\x10\x02\x42\x08\n\x06object2\xa7\x0b\n\x04\x44\x61ta\x12\xc4\x01\n\rExportIntents\x12\x33.zia.ai.playbook.data.v1alpha1.ExportIntentsRequest\x1a\x34.zia.ai.playbook.data.v1alpha1.ExportIntentsResponse\"H\x82\xd3\xe4\x93\x02\x42\"=/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/export:\x01*\x12\xad\x01\n\x11\x45xportIntentsHttp\x12\x33.zia.ai.playbook.data.v1alpha1.ExportIntentsRequest\x1a\x14.google.api.HttpBody\"M\x82\xd3\xe4\x93\x02G\"B/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/export_http:\x01*\x12\xc4\x01\n\rImportIntents\x12\x33.zia.ai.playbook.data.v1alpha1.ImportIntentsRequest\x1a\x34.zia.ai.playbook.data.v1alpha1.ImportIntentsResponse\"H\x82\xd3\xe4\x93\x02\x42\"=/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/import:\x01*\x12\xcd\x01\n\x11ImportIntentsHttp\x12\x33.zia.ai.playbook.data.v1alpha1.ImportIntentsRequest\x1a\x34.zia.ai.playbook.data.v1alpha1.ImportIntentsResponse\"M\x82\xd3\xe4\x93\x02G\"B/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/import_http:\x01*\x12\xd9\x01\n\x17ImportConversationsFile\x12=.zia.ai.playbook.data.v1alpha1.ImportConversationsFileRequest\x1a>.zia.ai.playbook.data.v1alpha1.ImportConversationsFileResponse\"?\x82\xd3\xe4\x93\x02\x39\"4/v1alpha1/files/{namespace}/{conversation_source_id}:\x01*\x12\xd0\x01\n\x15ListConversationsFile\x12;.zia.ai.playbook.data.v1alpha1.ListConversationsFileRequest\x1a<.zia.ai.playbook.data.v1alpha1.ListConversationsFileResponse\"<\x82\xd3\xe4\x93\x02\x36\x12\x34/v1alpha1/files/{namespace}/{conversation_source_id}\x12\xe1\x01\n\x17\x44\x65leteConversationsFile\x12=.zia.ai.playbook.data.v1alpha1.DeleteConversationsFileRequest\x1a>.zia.ai.playbook.data.v1alpha1.DeleteConversationsFileResponse\"G\x82\xd3\xe4\x93\x02\x41*?/v1alpha1/files/{namespace}/{conversation_source_id}/{filename}BIZGgithub.com/zia-ai/platform/pkg/api/playbook/data/v1alpha1;data_v1alpha1b\x06proto3')



_EXPORTINTENTSREQUEST = DESCRIPTOR.message_types_by_name['ExportIntentsRequest']
_EXPORTINTENTSRESPONSE = DESCRIPTOR.message_types_by_name['ExportIntentsResponse']
_IMPORTINTENTSREQUEST = DESCRIPTOR.message_types_by_name['ImportIntentsRequest']
_IMPORTINTENTSRESPONSE = DESCRIPTOR.message_types_by_name['ImportIntentsResponse']
_IMPORTCONVERSATIONSFILEREQUEST = DESCRIPTOR.message_types_by_name['ImportConversationsFileRequest']
_IMPORTCONVERSATIONSFILERESPONSE = DESCRIPTOR.message_types_by_name['ImportConversationsFileResponse']
_LISTCONVERSATIONSFILEREQUEST = DESCRIPTOR.message_types_by_name['ListConversationsFileRequest']
_LISTCONVERSATIONSFILERESPONSE = DESCRIPTOR.message_types_by_name['ListConversationsFileResponse']
_LISTCONVERSATIONSFILERESPONSE_FILE = _LISTCONVERSATIONSFILERESPONSE.nested_types_by_name['File']
_DELETECONVERSATIONSFILEREQUEST = DESCRIPTOR.message_types_by_name['DeleteConversationsFileRequest']
_DELETECONVERSATIONSFILERESPONSE = DESCRIPTOR.message_types_by_name['DeleteConversationsFileResponse']
_VALIDATIONPROBLEM = DESCRIPTOR.message_types_by_name['ValidationProblem']
_VALIDATIONPROBLEM_LEVEL = _VALIDATIONPROBLEM.enum_types_by_name['Level']
ExportIntentsRequest = _reflection.GeneratedProtocolMessageType('ExportIntentsRequest', (_message.Message,), {
  'DESCRIPTOR' : _EXPORTINTENTSREQUEST,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ExportIntentsRequest)
  })
_sym_db.RegisterMessage(ExportIntentsRequest)

ExportIntentsResponse = _reflection.GeneratedProtocolMessageType('ExportIntentsResponse', (_message.Message,), {
  'DESCRIPTOR' : _EXPORTINTENTSRESPONSE,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ExportIntentsResponse)
  })
_sym_db.RegisterMessage(ExportIntentsResponse)

ImportIntentsRequest = _reflection.GeneratedProtocolMessageType('ImportIntentsRequest', (_message.Message,), {
  'DESCRIPTOR' : _IMPORTINTENTSREQUEST,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ImportIntentsRequest)
  })
_sym_db.RegisterMessage(ImportIntentsRequest)

ImportIntentsResponse = _reflection.GeneratedProtocolMessageType('ImportIntentsResponse', (_message.Message,), {
  'DESCRIPTOR' : _IMPORTINTENTSRESPONSE,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ImportIntentsResponse)
  })
_sym_db.RegisterMessage(ImportIntentsResponse)

ImportConversationsFileRequest = _reflection.GeneratedProtocolMessageType('ImportConversationsFileRequest', (_message.Message,), {
  'DESCRIPTOR' : _IMPORTCONVERSATIONSFILEREQUEST,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ImportConversationsFileRequest)
  })
_sym_db.RegisterMessage(ImportConversationsFileRequest)

ImportConversationsFileResponse = _reflection.GeneratedProtocolMessageType('ImportConversationsFileResponse', (_message.Message,), {
  'DESCRIPTOR' : _IMPORTCONVERSATIONSFILERESPONSE,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ImportConversationsFileResponse)
  })
_sym_db.RegisterMessage(ImportConversationsFileResponse)

ListConversationsFileRequest = _reflection.GeneratedProtocolMessageType('ListConversationsFileRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTCONVERSATIONSFILEREQUEST,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ListConversationsFileRequest)
  })
_sym_db.RegisterMessage(ListConversationsFileRequest)

ListConversationsFileResponse = _reflection.GeneratedProtocolMessageType('ListConversationsFileResponse', (_message.Message,), {

  'File' : _reflection.GeneratedProtocolMessageType('File', (_message.Message,), {
    'DESCRIPTOR' : _LISTCONVERSATIONSFILERESPONSE_FILE,
    '__module__' : 'playbook.data.v1alpha1.data_pb2'
    # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ListConversationsFileResponse.File)
    })
  ,
  'DESCRIPTOR' : _LISTCONVERSATIONSFILERESPONSE,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ListConversationsFileResponse)
  })
_sym_db.RegisterMessage(ListConversationsFileResponse)
_sym_db.RegisterMessage(ListConversationsFileResponse.File)

DeleteConversationsFileRequest = _reflection.GeneratedProtocolMessageType('DeleteConversationsFileRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETECONVERSATIONSFILEREQUEST,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.DeleteConversationsFileRequest)
  })
_sym_db.RegisterMessage(DeleteConversationsFileRequest)

DeleteConversationsFileResponse = _reflection.GeneratedProtocolMessageType('DeleteConversationsFileResponse', (_message.Message,), {
  'DESCRIPTOR' : _DELETECONVERSATIONSFILERESPONSE,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.DeleteConversationsFileResponse)
  })
_sym_db.RegisterMessage(DeleteConversationsFileResponse)

ValidationProblem = _reflection.GeneratedProtocolMessageType('ValidationProblem', (_message.Message,), {
  'DESCRIPTOR' : _VALIDATIONPROBLEM,
  '__module__' : 'playbook.data.v1alpha1.data_pb2'
  # @@protoc_insertion_point(class_scope:zia.ai.playbook.data.v1alpha1.ValidationProblem)
  })
_sym_db.RegisterMessage(ValidationProblem)

_DATA = DESCRIPTOR.services_by_name['Data']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZGgithub.com/zia-ai/platform/pkg/api/playbook/data/v1alpha1;data_v1alpha1'
  _IMPORTINTENTSREQUEST.fields_by_name['clear_workspace']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['clear_workspace']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['clear_intents']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['clear_intents']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['clear_entities']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['clear_entities']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['clear_tags']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['clear_tags']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['merge_intents']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['merge_intents']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['merge_entities']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['merge_entities']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['merge_tags']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['merge_tags']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['extra_intent_tags']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['extra_intent_tags']._serialized_options = b'\030\001'
  _IMPORTINTENTSREQUEST.fields_by_name['extra_phrase_tags']._options = None
  _IMPORTINTENTSREQUEST.fields_by_name['extra_phrase_tags']._serialized_options = b'\030\001'
  _DATA.methods_by_name['ExportIntents']._options = None
  _DATA.methods_by_name['ExportIntents']._serialized_options = b'\202\323\344\223\002B\"=/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/export:\001*'
  _DATA.methods_by_name['ExportIntentsHttp']._options = None
  _DATA.methods_by_name['ExportIntentsHttp']._serialized_options = b'\202\323\344\223\002G\"B/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/export_http:\001*'
  _DATA.methods_by_name['ImportIntents']._options = None
  _DATA.methods_by_name['ImportIntents']._serialized_options = b'\202\323\344\223\002B\"=/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/import:\001*'
  _DATA.methods_by_name['ImportIntentsHttp']._options = None
  _DATA.methods_by_name['ImportIntentsHttp']._serialized_options = b'\202\323\344\223\002G\"B/v1alpha1/workspaces/{namespace}/{playbook_id}/intents/import_http:\001*'
  _DATA.methods_by_name['ImportConversationsFile']._options = None
  _DATA.methods_by_name['ImportConversationsFile']._serialized_options = b'\202\323\344\223\0029\"4/v1alpha1/files/{namespace}/{conversation_source_id}:\001*'
  _DATA.methods_by_name['ListConversationsFile']._options = None
  _DATA.methods_by_name['ListConversationsFile']._serialized_options = b'\202\323\344\223\0026\0224/v1alpha1/files/{namespace}/{conversation_source_id}'
  _DATA.methods_by_name['DeleteConversationsFile']._options = None
  _DATA.methods_by_name['DeleteConversationsFile']._serialized_options = b'\202\323\344\223\002A*?/v1alpha1/files/{namespace}/{conversation_source_id}/{filename}'
  _EXPORTINTENTSREQUEST._serialized_start=309
  _EXPORTINTENTSREQUEST._serialized_end=546
  _EXPORTINTENTSRESPONSE._serialized_start=549
  _EXPORTINTENTSRESPONSE._serialized_end=678
  _IMPORTINTENTSREQUEST._serialized_start=681
  _IMPORTINTENTSREQUEST._serialized_end=1311
  _IMPORTINTENTSRESPONSE._serialized_start=1314
  _IMPORTINTENTSRESPONSE._serialized_end=1570
  _IMPORTCONVERSATIONSFILEREQUEST._serialized_start=1573
  _IMPORTCONVERSATIONSFILEREQUEST._serialized_end=1764
  _IMPORTCONVERSATIONSFILERESPONSE._serialized_start=1767
  _IMPORTCONVERSATIONSFILERESPONSE._serialized_end=1910
  _LISTCONVERSATIONSFILEREQUEST._serialized_start=1912
  _LISTCONVERSATIONSFILEREQUEST._serialized_end=1993
  _LISTCONVERSATIONSFILERESPONSE._serialized_start=1996
  _LISTCONVERSATIONSFILERESPONSE._serialized_end=2264
  _LISTCONVERSATIONSFILERESPONSE_FILE._serialized_start=2112
  _LISTCONVERSATIONSFILERESPONSE_FILE._serialized_end=2264
  _DELETECONVERSATIONSFILEREQUEST._serialized_start=2266
  _DELETECONVERSATIONSFILEREQUEST._serialized_end=2367
  _DELETECONVERSATIONSFILERESPONSE._serialized_start=2369
  _DELETECONVERSATIONSFILERESPONSE._serialized_end=2402
  _VALIDATIONPROBLEM._serialized_start=2405
  _VALIDATIONPROBLEM._serialized_end=2660
  _VALIDATIONPROBLEM_LEVEL._serialized_start=2606
  _VALIDATIONPROBLEM_LEVEL._serialized_end=2650
  _DATA._serialized_start=2663
  _DATA._serialized_end=4110
# @@protoc_insertion_point(module_scope)
