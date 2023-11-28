import sys
from concurrent import futures 
import grpc
import numpy as np
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1 import health
# import generated gRPC stubs
from grpc_generated import services_pb2_grpc,services_pb2

class NumberSortingService(services_pb2_grpc.NumberSortingService):
  def SortNumbers(self,request,context):
        arr = np.array(request.numbers)
        result = np.sort(arr)
        print(f"Sorted {len(result)} numbers")
        return services_pb2.NumberArray(numbers=result)
  
class UrMumJoke(services_pb2_grpc.UrMumJoke):
  def TellJoke(self, request, context):
      return services_pb2.theJokeReply(message="Ur mum is so, %s, she %s !" % (request.reason, request.punchline))
def serve():
  DEFAULT_PORT = 50055
  # Get the port number from the command line parameter    
  port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
  HOST = f'localhost:{port}'

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

  # TODO, add your gRPC service to self-hosted server, e.g.
  services_pb2_grpc.add_NumberSortingServiceServicer_to_server(NumberSortingService(), server)
  services_pb2_grpc.add_UrMumJokeServicer_to_server(UrMumJoke(),server)

  health_pb2_grpc.add_HealthServicer_to_server(health.HealthServicer(), server)
  server.add_insecure_port(HOST)
  print(f"gRPC server started and listening on {HOST}")
  server.start()
  server.wait_for_termination()
  
if __name__ == '__main__':
  serve()  

#  python server/server.py 
# python -m grpc_tools.protoc -I../../protos --python_out=../helloworld2 --pyi_out=../helloworld2 --grpc_python_out=../helloworld2 ../../protos/helloworld.proto
# python -m grpc_tools.protoc -I../../protos --python_out=../helloworld2 --pyi_out=../helloworld2 --grpc_python_out=../helloworld2 ../../protos/helloworld.proto
# python -m grpc_tools.protoc -I./protos --python_out=./server/grpc_generated --grpc_python_out=./server/grpc_generated ./protos/services.proto