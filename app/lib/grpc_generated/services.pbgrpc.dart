//
//  Generated code. Do not modify.
//  source: services.proto
//
// @dart = 2.12

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_final_fields
// ignore_for_file: unnecessary_import, unnecessary_this, unused_import

import 'dart:async' as $async;
import 'dart:core' as $core;

import 'package:grpc/service_api.dart' as $grpc;
import 'package:protobuf/protobuf.dart' as $pb;

import 'services.pb.dart' as $0;

export 'services.pb.dart';

@$pb.GrpcServiceName('NumberSortingService')
class NumberSortingServiceClient extends $grpc.Client {
  static final _$sortNumbers = $grpc.ClientMethod<$0.NumberArray, $0.NumberArray>(
      '/NumberSortingService/SortNumbers',
      ($0.NumberArray value) => value.writeToBuffer(),
      ($core.List<$core.int> value) => $0.NumberArray.fromBuffer(value));

  NumberSortingServiceClient($grpc.ClientChannel channel,
      {$grpc.CallOptions? options,
      $core.Iterable<$grpc.ClientInterceptor>? interceptors})
      : super(channel, options: options,
        interceptors: interceptors);

  $grpc.ResponseFuture<$0.NumberArray> sortNumbers($0.NumberArray request, {$grpc.CallOptions? options}) {
    return $createUnaryCall(_$sortNumbers, request, options: options);
  }
}

@$pb.GrpcServiceName('NumberSortingService')
abstract class NumberSortingServiceBase extends $grpc.Service {
  $core.String get $name => 'NumberSortingService';

  NumberSortingServiceBase() {
    $addMethod($grpc.ServiceMethod<$0.NumberArray, $0.NumberArray>(
        'SortNumbers',
        sortNumbers_Pre,
        false,
        false,
        ($core.List<$core.int> value) => $0.NumberArray.fromBuffer(value),
        ($0.NumberArray value) => value.writeToBuffer()));
  }

  $async.Future<$0.NumberArray> sortNumbers_Pre($grpc.ServiceCall call, $async.Future<$0.NumberArray> request) async {
    return sortNumbers(call, await request);
  }

  $async.Future<$0.NumberArray> sortNumbers($grpc.ServiceCall call, $0.NumberArray request);
}
@$pb.GrpcServiceName('UrMumJoke')
class UrMumJokeClient extends $grpc.Client {
  static final _$tellJoke = $grpc.ClientMethod<$0.theJokeReq, $0.theJokeReply>(
      '/UrMumJoke/TellJoke',
      ($0.theJokeReq value) => value.writeToBuffer(),
      ($core.List<$core.int> value) => $0.theJokeReply.fromBuffer(value));

  UrMumJokeClient($grpc.ClientChannel channel,
      {$grpc.CallOptions? options,
      $core.Iterable<$grpc.ClientInterceptor>? interceptors})
      : super(channel, options: options,
        interceptors: interceptors);

  $grpc.ResponseFuture<$0.theJokeReply> tellJoke($0.theJokeReq request, {$grpc.CallOptions? options}) {
    return $createUnaryCall(_$tellJoke, request, options: options);
  }
}

@$pb.GrpcServiceName('UrMumJoke')
abstract class UrMumJokeServiceBase extends $grpc.Service {
  $core.String get $name => 'UrMumJoke';

  UrMumJokeServiceBase() {
    $addMethod($grpc.ServiceMethod<$0.theJokeReq, $0.theJokeReply>(
        'TellJoke',
        tellJoke_Pre,
        false,
        false,
        ($core.List<$core.int> value) => $0.theJokeReq.fromBuffer(value),
        ($0.theJokeReply value) => value.writeToBuffer()));
  }

  $async.Future<$0.theJokeReply> tellJoke_Pre($grpc.ServiceCall call, $async.Future<$0.theJokeReq> request) async {
    return tellJoke(call, await request);
  }

  $async.Future<$0.theJokeReply> tellJoke($grpc.ServiceCall call, $0.theJokeReq request);
}
@$pb.GrpcServiceName('EMGClassifierService')
class EMGClassifierServiceClient extends $grpc.Client {
  static final _$classify_Signal = $grpc.ClientMethod<$0.PredictRequest, $0.PredictedSignal>(
      '/EMGClassifierService/Classify_Signal',
      ($0.PredictRequest value) => value.writeToBuffer(),
      ($core.List<$core.int> value) => $0.PredictedSignal.fromBuffer(value));

  EMGClassifierServiceClient($grpc.ClientChannel channel,
      {$grpc.CallOptions? options,
      $core.Iterable<$grpc.ClientInterceptor>? interceptors})
      : super(channel, options: options,
        interceptors: interceptors);

  $grpc.ResponseFuture<$0.PredictedSignal> classify_Signal($0.PredictRequest request, {$grpc.CallOptions? options}) {
    return $createUnaryCall(_$classify_Signal, request, options: options);
  }
}

@$pb.GrpcServiceName('EMGClassifierService')
abstract class EMGClassifierServiceBase extends $grpc.Service {
  $core.String get $name => 'EMGClassifierService';

  EMGClassifierServiceBase() {
    $addMethod($grpc.ServiceMethod<$0.PredictRequest, $0.PredictedSignal>(
        'Classify_Signal',
        classify_Signal_Pre,
        false,
        false,
        ($core.List<$core.int> value) => $0.PredictRequest.fromBuffer(value),
        ($0.PredictedSignal value) => value.writeToBuffer()));
  }

  $async.Future<$0.PredictedSignal> classify_Signal_Pre($grpc.ServiceCall call, $async.Future<$0.PredictRequest> request) async {
    return classify_Signal(call, await request);
  }

  $async.Future<$0.PredictedSignal> classify_Signal($grpc.ServiceCall call, $0.PredictRequest request);
}
@$pb.GrpcServiceName('PoseHandService')
class PoseHandServiceClient extends $grpc.Client {
  static final _$poseHand = $grpc.ClientMethod<$0.PredictedSignal, $0.PlaceHolderMsg>(
      '/PoseHandService/PoseHand',
      ($0.PredictedSignal value) => value.writeToBuffer(),
      ($core.List<$core.int> value) => $0.PlaceHolderMsg.fromBuffer(value));

  PoseHandServiceClient($grpc.ClientChannel channel,
      {$grpc.CallOptions? options,
      $core.Iterable<$grpc.ClientInterceptor>? interceptors})
      : super(channel, options: options,
        interceptors: interceptors);

  $grpc.ResponseFuture<$0.PlaceHolderMsg> poseHand($0.PredictedSignal request, {$grpc.CallOptions? options}) {
    return $createUnaryCall(_$poseHand, request, options: options);
  }
}

@$pb.GrpcServiceName('PoseHandService')
abstract class PoseHandServiceBase extends $grpc.Service {
  $core.String get $name => 'PoseHandService';

  PoseHandServiceBase() {
    $addMethod($grpc.ServiceMethod<$0.PredictedSignal, $0.PlaceHolderMsg>(
        'PoseHand',
        poseHand_Pre,
        false,
        false,
        ($core.List<$core.int> value) => $0.PredictedSignal.fromBuffer(value),
        ($0.PlaceHolderMsg value) => value.writeToBuffer()));
  }

  $async.Future<$0.PlaceHolderMsg> poseHand_Pre($grpc.ServiceCall call, $async.Future<$0.PredictedSignal> request) async {
    return poseHand(call, await request);
  }

  $async.Future<$0.PlaceHolderMsg> poseHand($grpc.ServiceCall call, $0.PredictedSignal request);
}
