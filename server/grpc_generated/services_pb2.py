# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: services.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0eservices.proto\"\x1e\n\x0bNumberArray\x12\x0f\n\x07numbers\x18\x01 \x03(\x05\"/\n\ntheJokeReq\x12\x11\n\tpunchline\x18\x01 \x01(\t\x12\x0e\n\x06reason\x18\x02 \x01(\t\"\x1f\n\x0ctheJokeReply\x12\x0f\n\x07message\x18\x01 \x01(\t\"5\n\x0fPredictedSignal\x12\x0e\n\x06signal\x18\x01 \x01(\x05\x12\x12\n\nbase64plot\x18\x02 \x01(\t\"\x10\n\x0ePlaceHolderMsg\"2\n\x0ePredictRequest\x12\x10\n\x08idx_from\x18\x01 \x01(\x05\x12\x0e\n\x06idx_to\x18\x02 \x01(\x05\x32\x43\n\x14NumberSortingService\x12+\n\x0bSortNumbers\x12\x0c.NumberArray\x1a\x0c.NumberArray\"\x00\x32\x35\n\tUrMumJoke\x12(\n\x08TellJoke\x12\x0b.theJokeReq\x1a\r.theJokeReply\"\x00\x32N\n\x14\x45MGClassifierService\x12\x36\n\x0f\x43lassify_Signal\x12\x0f.PredictRequest\x1a\x10.PredictedSignal\"\x00\x32\x42\n\x0fPoseHandService\x12/\n\x08PoseHand\x12\x10.PredictedSignal\x1a\x0f.PlaceHolderMsg\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'services_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_NUMBERARRAY']._serialized_start=18
  _globals['_NUMBERARRAY']._serialized_end=48
  _globals['_THEJOKEREQ']._serialized_start=50
  _globals['_THEJOKEREQ']._serialized_end=97
  _globals['_THEJOKEREPLY']._serialized_start=99
  _globals['_THEJOKEREPLY']._serialized_end=130
  _globals['_PREDICTEDSIGNAL']._serialized_start=132
  _globals['_PREDICTEDSIGNAL']._serialized_end=185
  _globals['_PLACEHOLDERMSG']._serialized_start=187
  _globals['_PLACEHOLDERMSG']._serialized_end=203
  _globals['_PREDICTREQUEST']._serialized_start=205
  _globals['_PREDICTREQUEST']._serialized_end=255
  _globals['_NUMBERSORTINGSERVICE']._serialized_start=257
  _globals['_NUMBERSORTINGSERVICE']._serialized_end=324
  _globals['_URMUMJOKE']._serialized_start=326
  _globals['_URMUMJOKE']._serialized_end=379
  _globals['_EMGCLASSIFIERSERVICE']._serialized_start=381
  _globals['_EMGCLASSIFIERSERVICE']._serialized_end=459
  _globals['_POSEHANDSERVICE']._serialized_start=461
  _globals['_POSEHANDSERVICE']._serialized_end=527
# @@protoc_insertion_point(module_scope)
