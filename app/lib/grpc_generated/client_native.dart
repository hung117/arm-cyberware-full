import 'package:grpc/grpc.dart';
import 'package:grpc/grpc_connection_interface.dart';

ClientChannelBase getGrpcClientChannel(
    String host, int port, bool useHttps, bool isAndroid) {
  host = isAndroid ? "10.0.2.2" : "localhost";
  final channel = ClientChannel(
    host,
    port: 50055,
    options: const ChannelOptions(credentials: ChannelCredentials.insecure()),
  );
  return channel;
}
