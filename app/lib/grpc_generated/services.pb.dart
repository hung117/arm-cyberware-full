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
  factory NumberArray.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory NumberArray.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'NumberArray',
      createEmptyInstance: create)
    ..p<$core.int>(1, _omitFieldNames ? '' : 'numbers', $pb.PbFieldType.K3)
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  NumberArray clone() => NumberArray()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  NumberArray copyWith(void Function(NumberArray) updates) =>
      super.copyWith((message) => updates(message as NumberArray))
          as NumberArray;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static NumberArray create() => NumberArray._();
  NumberArray createEmptyInstance() => create();
  static $pb.PbList<NumberArray> createRepeated() => $pb.PbList<NumberArray>();
  @$core.pragma('dart2js:noInline')
  static NumberArray getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<NumberArray>(create);
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
  factory theJokeReq.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory theJokeReq.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'theJokeReq',
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'punchline')
    ..aOS(2, _omitFieldNames ? '' : 'reason')
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  theJokeReq clone() => theJokeReq()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  theJokeReq copyWith(void Function(theJokeReq) updates) =>
      super.copyWith((message) => updates(message as theJokeReq)) as theJokeReq;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static theJokeReq create() => theJokeReq._();
  theJokeReq createEmptyInstance() => create();
  static $pb.PbList<theJokeReq> createRepeated() => $pb.PbList<theJokeReq>();
  @$core.pragma('dart2js:noInline')
  static theJokeReq getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<theJokeReq>(create);
  static theJokeReq? _defaultInstance;

  @$pb.TagNumber(1)
  $core.String get punchline => $_getSZ(0);
  @$pb.TagNumber(1)
  set punchline($core.String v) {
    $_setString(0, v);
  }

  @$pb.TagNumber(1)
  $core.bool hasPunchline() => $_has(0);
  @$pb.TagNumber(1)
  void clearPunchline() => clearField(1);

  @$pb.TagNumber(2)
  $core.String get reason => $_getSZ(1);
  @$pb.TagNumber(2)
  set reason($core.String v) {
    $_setString(1, v);
  }

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
  factory theJokeReply.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory theJokeReply.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'theJokeReply',
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'message')
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  theJokeReply clone() => theJokeReply()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  theJokeReply copyWith(void Function(theJokeReply) updates) =>
      super.copyWith((message) => updates(message as theJokeReply))
          as theJokeReply;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static theJokeReply create() => theJokeReply._();
  theJokeReply createEmptyInstance() => create();
  static $pb.PbList<theJokeReply> createRepeated() =>
      $pb.PbList<theJokeReply>();
  @$core.pragma('dart2js:noInline')
  static theJokeReply getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<theJokeReply>(create);
  static theJokeReply? _defaultInstance;

  @$pb.TagNumber(1)
  $core.String get message => $_getSZ(0);
  @$pb.TagNumber(1)
  set message($core.String v) {
    $_setString(0, v);
  }

  @$pb.TagNumber(1)
  $core.bool hasMessage() => $_has(0);
  @$pb.TagNumber(1)
  void clearMessage() => clearField(1);
}

class PredictedSignal extends $pb.GeneratedMessage {
  factory PredictedSignal({
    $core.int? signal,
    $core.String? base64plot,
  }) {
    final $result = create();
    if (signal != null) {
      $result.signal = signal;
    }
    if (base64plot != null) {
      $result.base64plot = base64plot;
    }
    return $result;
  }
  PredictedSignal._() : super();
  factory PredictedSignal.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory PredictedSignal.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'PredictedSignal',
      createEmptyInstance: create)
    ..a<$core.int>(1, _omitFieldNames ? '' : 'signal', $pb.PbFieldType.O3)
    ..aOS(2, _omitFieldNames ? '' : 'base64plot')
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  PredictedSignal clone() => PredictedSignal()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  PredictedSignal copyWith(void Function(PredictedSignal) updates) =>
      super.copyWith((message) => updates(message as PredictedSignal))
          as PredictedSignal;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static PredictedSignal create() => PredictedSignal._();
  PredictedSignal createEmptyInstance() => create();
  static $pb.PbList<PredictedSignal> createRepeated() =>
      $pb.PbList<PredictedSignal>();
  @$core.pragma('dart2js:noInline')
  static PredictedSignal getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<PredictedSignal>(create);
  static PredictedSignal? _defaultInstance;

  @$pb.TagNumber(1)
  $core.int get signal => $_getIZ(0);
  @$pb.TagNumber(1)
  set signal($core.int v) {
    $_setSignedInt32(0, v);
  }

  @$pb.TagNumber(1)
  $core.bool hasSignal() => $_has(0);
  @$pb.TagNumber(1)
  void clearSignal() => clearField(1);

  @$pb.TagNumber(2)
  $core.String get base64plot => $_getSZ(1);
  @$pb.TagNumber(2)
  set base64plot($core.String v) {
    $_setString(1, v);
  }

  @$pb.TagNumber(2)
  $core.bool hasBase64plot() => $_has(1);
  @$pb.TagNumber(2)
  void clearBase64plot() => clearField(2);
}

class PlaceHolderMsg extends $pb.GeneratedMessage {
  factory PlaceHolderMsg() => create();
  PlaceHolderMsg._() : super();
  factory PlaceHolderMsg.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory PlaceHolderMsg.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'PlaceHolderMsg',
      createEmptyInstance: create)
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  PlaceHolderMsg clone() => PlaceHolderMsg()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  PlaceHolderMsg copyWith(void Function(PlaceHolderMsg) updates) =>
      super.copyWith((message) => updates(message as PlaceHolderMsg))
          as PlaceHolderMsg;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static PlaceHolderMsg create() => PlaceHolderMsg._();
  PlaceHolderMsg createEmptyInstance() => create();
  static $pb.PbList<PlaceHolderMsg> createRepeated() =>
      $pb.PbList<PlaceHolderMsg>();
  @$core.pragma('dart2js:noInline')
  static PlaceHolderMsg getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<PlaceHolderMsg>(create);
  static PlaceHolderMsg? _defaultInstance;
}

class PredictRequest extends $pb.GeneratedMessage {
  factory PredictRequest({
    $core.int? idxFrom,
    $core.int? idxTestPose,
    $core.int? idxUser,
  }) {
    final $result = create();
    if (idxFrom != null) {
      $result.idxFrom = idxFrom;
    }
    if (idxTestPose != null) {
      $result.idxTestPose = idxTestPose;
    }
    if (idxUser != null) {
      $result.idxUser = idxUser;
    }
    return $result;
  }
  PredictRequest._() : super();
  factory PredictRequest.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory PredictRequest.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'PredictRequest',
      createEmptyInstance: create)
    ..a<$core.int>(1, _omitFieldNames ? '' : 'idxFrom', $pb.PbFieldType.O3)
    ..a<$core.int>(2, _omitFieldNames ? '' : 'idxTestPose', $pb.PbFieldType.O3)
    ..a<$core.int>(3, _omitFieldNames ? '' : 'idxUser', $pb.PbFieldType.O3)
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  PredictRequest clone() => PredictRequest()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  PredictRequest copyWith(void Function(PredictRequest) updates) =>
      super.copyWith((message) => updates(message as PredictRequest))
          as PredictRequest;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static PredictRequest create() => PredictRequest._();
  PredictRequest createEmptyInstance() => create();
  static $pb.PbList<PredictRequest> createRepeated() =>
      $pb.PbList<PredictRequest>();
  @$core.pragma('dart2js:noInline')
  static PredictRequest getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<PredictRequest>(create);
  static PredictRequest? _defaultInstance;

  @$pb.TagNumber(1)
  $core.int get idxFrom => $_getIZ(0);
  @$pb.TagNumber(1)
  set idxFrom($core.int v) {
    $_setSignedInt32(0, v);
  }

  @$pb.TagNumber(1)
  $core.bool hasIdxFrom() => $_has(0);
  @$pb.TagNumber(1)
  void clearIdxFrom() => clearField(1);

  @$pb.TagNumber(2)
  $core.int get idxTestPose => $_getIZ(1);
  @$pb.TagNumber(2)
  set idxTestPose($core.int v) {
    $_setSignedInt32(1, v);
  }

  @$pb.TagNumber(2)
  $core.bool hasIdxTestPose() => $_has(1);
  @$pb.TagNumber(2)
  void clearIdxTestPose() => clearField(2);

  @$pb.TagNumber(3)
  $core.int get idxUser => $_getIZ(2);
  @$pb.TagNumber(3)
  set idxUser($core.int v) {
    $_setSignedInt32(2, v);
  }

  @$pb.TagNumber(3)
  $core.bool hasIdxUser() => $_has(2);
  @$pb.TagNumber(3)
  void clearIdxUser() => clearField(3);
}

class PoseRequest extends $pb.GeneratedMessage {
  factory PoseRequest({
    $core.int? pose,
  }) {
    final $result = create();
    if (pose != null) {
      $result.pose = pose;
    }
    return $result;
  }
  PoseRequest._() : super();
  factory PoseRequest.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory PoseRequest.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'PoseRequest',
      createEmptyInstance: create)
    ..a<$core.int>(1, _omitFieldNames ? '' : 'pose', $pb.PbFieldType.O3)
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  PoseRequest clone() => PoseRequest()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  PoseRequest copyWith(void Function(PoseRequest) updates) =>
      super.copyWith((message) => updates(message as PoseRequest))
          as PoseRequest;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static PoseRequest create() => PoseRequest._();
  PoseRequest createEmptyInstance() => create();
  static $pb.PbList<PoseRequest> createRepeated() => $pb.PbList<PoseRequest>();
  @$core.pragma('dart2js:noInline')
  static PoseRequest getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<PoseRequest>(create);
  static PoseRequest? _defaultInstance;

  @$pb.TagNumber(1)
  $core.int get pose => $_getIZ(0);
  @$pb.TagNumber(1)
  set pose($core.int v) {
    $_setSignedInt32(0, v);
  }

  @$pb.TagNumber(1)
  $core.bool hasPose() => $_has(0);
  @$pb.TagNumber(1)
  void clearPose() => clearField(1);
}

class PoseHand extends $pb.GeneratedMessage {
  factory PoseHand() => create();
  PoseHand._() : super();
  factory PoseHand.fromBuffer($core.List<$core.int> i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(i, r);
  factory PoseHand.fromJson($core.String i,
          [$pb.ExtensionRegistry r = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(i, r);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'PoseHand',
      createEmptyInstance: create)
    ..hasRequiredFields = false;

  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.deepCopy] instead. '
      'Will be removed in next major version')
  PoseHand clone() => PoseHand()..mergeFromMessage(this);
  @$core.Deprecated('Using this can add significant overhead to your binary. '
      'Use [GeneratedMessageGenericExtensions.rebuild] instead. '
      'Will be removed in next major version')
  PoseHand copyWith(void Function(PoseHand) updates) =>
      super.copyWith((message) => updates(message as PoseHand)) as PoseHand;

  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static PoseHand create() => PoseHand._();
  PoseHand createEmptyInstance() => create();
  static $pb.PbList<PoseHand> createRepeated() => $pb.PbList<PoseHand>();
  @$core.pragma('dart2js:noInline')
  static PoseHand getDefault() =>
      _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<PoseHand>(create);
  static PoseHand? _defaultInstance;
}

const _omitFieldNames = $core.bool.fromEnvironment('protobuf.omit_field_names');
const _omitMessageNames =
    $core.bool.fromEnvironment('protobuf.omit_message_names');
