//
//  Generated code. Do not modify.
//  source: services.proto
//
// @dart = 2.12

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_final_fields
// ignore_for_file: unnecessary_import, unnecessary_this, unused_import

import 'dart:core' as $core;

import 'package:protobuf/protobuf.dart' as $pb;

class NumberArray extends $pb.GeneratedMessage {
  factory NumberArray({
    $core.Iterable<$core.int>? numbers,
  }) {
    final $result = create();
    if (numbers != null) {
      $result.numbers.addAll(numbers);
    }
    return $result;
  }
  NumberArray._() : super();
  factory NumberArray.fromBuffer($core.List<$core.int> i, [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) => create()..mergeFromBuffer(i, r);
  factory NumberArray.fromJson($core.String i, [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) => create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(_omitMessageNames ? '' : 'NumberArray', createEmptyInstance: create)
    ..p<$core.int>(1, _omitFieldNames ? '' : 'numbers', $pb.PbFieldType.K3)
    ..hasRequiredFields = false
  ;

  @$core.Deprecated(
  'Using this can add significant overhead to your binary. '
  'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
  'Will be removed in next major version')
  NumberArray clone() => NumberArray()..mergeFromMessage(this);
  @$core.Deprecated(
  'Using this can add significant overhead to your binary. '
  'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
  'Will be removed in next major version')
  NumberArray copyWith(void Function(NumberArray) updates) => super.copyWith((message) => updates(message as NumberArray)) as NumberArray;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static NumberArray create() => NumberArray._();
  NumberArray createEmptyInstance() => create();
  static $pb.PbList<NumberArray> createRepeated() => $pb.PbList<NumberArray>();
  @$core.pragma('dart2js:noInline')
  static NumberArray getDefault() => _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<NumberArray>(create);
  static NumberArray? _defaultInstance;

  @$pb.TagNumber(1)
  $core.List<$core.int> get numbers => $_getList(0);
}

class theJokeReq extends $pb.GeneratedMessage {
  factory theJokeReq({
    $core.String? punchline,
    $core.String? reason,
  }) {
    final $result = create();
    if (punchline != null) {
      $result.punchline = punchline;
    }
    if (reason != null) {
      $result.reason = reason;
    }
    return $result;
  }
  theJokeReq._() : super();
  factory theJokeReq.fromBuffer($core.List<$core.int> i, [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) => create()..mergeFromBuffer(i, r);
  factory theJokeReq.fromJson($core.String i, [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) => create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(_omitMessageNames ? '' : 'theJokeReq', createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'punchline')
    ..aOS(2, _omitFieldNames ? '' : 'reason')
    ..hasRequiredFields = false
  ;

  @$core.Deprecated(
  'Using this can add significant overhead to your binary. '
  'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
  'Will be removed in next major version')
  theJokeReq clone() => theJokeReq()..mergeFromMessage(this);
  @$core.Deprecated(
  'Using this can add significant overhead to your binary. '
  'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
  'Will be removed in next major version')
  theJokeReq copyWith(void Function(theJokeReq) updates) => super.copyWith((message) => updates(message as theJokeReq)) as theJokeReq;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static theJokeReq create() => theJokeReq._();
  theJokeReq createEmptyInstance() => create();
  static $pb.PbList<theJokeReq> createRepeated() => $pb.PbList<theJokeReq>();
  @$core.pragma('dart2js:noInline')
  static theJokeReq getDefault() => _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<theJokeReq>(create);
  static theJokeReq? _defaultInstance;

  @$pb.TagNumber(1)
  $core.String get punchline => $_getSZ(0);
  @$pb.TagNumber(1)
  set punchline($core.String v) { $_setString(0, v); }
  @$pb.TagNumber(1)
  $core.bool hasPunchline() => $_has(0);
  @$pb.TagNumber(1)
  void clearPunchline() => clearField(1);

  @$pb.TagNumber(2)
  $core.String get reason => $_getSZ(1);
  @$pb.TagNumber(2)
  set reason($core.String v) { $_setString(1, v); }
  @$pb.TagNumber(2)
  $core.bool hasReason() => $_has(1);
  @$pb.TagNumber(2)
  void clearReason() => clearField(2);
}

class theJokeReply extends $pb.GeneratedMessage {
  factory theJokeReply({
    $core.String? message,
  }) {
    final $result = create();
    if (message != null) {
      $result.message = message;
    }
    return $result;
  }
  theJokeReply._() : super();
  factory theJokeReply.fromBuffer($core.List<$core.int> i, [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) => create()..mergeFromBuffer(i, r);
  factory theJokeReply.fromJson($core.String i, [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) => create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(_omitMessageNames ? '' : 'theJokeReply', createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'message')
    ..hasRequiredFields = false
  ;

  @$core.Deprecated(
  'Using this can add significant overhead to your binary. '
  'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
  'Will be removed in next major version')
  theJokeReply clone() => theJokeReply()..mergeFromMessage(this);
  @$core.Deprecated(
  'Using this can add significant overhead to your binary. '
  'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
  'Will be removed in next major version')
  theJokeReply copyWith(void Function(theJokeReply) updates) => super.copyWith((message) => updates(message as theJokeReply)) as theJokeReply;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static theJokeReply create() => theJokeReply._();
  theJokeReply createEmptyInstance() => create();
  static $pb.PbList<theJokeReply> createRepeated() => $pb.PbList<theJokeReply>();
  @$core.pragma('dart2js:noInline')
  static theJokeReply getDefault() => _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<theJokeReply>(create);
  static theJokeReply? _defaultInstance;

  @$pb.TagNumber(1)
  $core.String get message => $_getSZ(0);
  @$pb.TagNumber(1)
  set message($core.String v) { $_setString(0, v); }
  @$pb.TagNumber(1)
  $core.bool hasMessage() => $_has(0);
  @$pb.TagNumber(1)
  void clearMessage() => clearField(1);
}


const _omitFieldNames = $core.bool.fromEnvironment('protobuf.omit_field_names');
const _omitMessageNames = $core.bool.fromEnvironment('protobuf.omit_message_names');
