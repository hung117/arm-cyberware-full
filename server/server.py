import sys
from concurrent import futures 
import grpc
import numpy as np
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1 import health
# import generated gRPC stubs
from grpc_generated import services_pb2_grpc,services_pb2

import classifier
class NumberSortingService(services_pb2_grpc.NumberSortingService):
  def SortNumbers(self,request,context):
        arr = np.array(request.numbers)
        result = np.sort(arr)
        print(f"Sorted {len(result)} numbers")
        return services_pb2.NumberArray(numbers=result)
  
class UrMumJoke(services_pb2_grpc.UrMumJoke):
  def TellJoke(self, request, context):
      return services_pb2.theJokeReply(message="Ur mum is so, %s, she %s !" % (request.reason, request.punchline))

class EMGClassifierService(services_pb2_grpc.EMGClassifierService):
  def Classify_Signal(self, request, context):
      print("request data from %s, test pose %s, user request: %s"%(request.idx_from,request.idx_test_pose,request.idx_user))
      # predplot = classifier.predict_plot()
      # predplot = classifier.predict_plot(request.idx_from,request.idx_to) # the to is now act as the wanted test pose, will fix/clean later
      # predplot = classifier.predict_plot(request.idx_from,request.idx_to) # the to is now act as the wanted test pose, will fix/clean later
      predplot = classifier.predict_plot(request.idx_from,request.idx_test_pose,request.idx_user) # the to is now act as the wanted test pose, will fix/clean later
      
      iPose = predplot['pred']
      base64 = predplot['base64']
      print("base64 after pred: %s"%base64)
      
      return services_pb2.PredictedSignal(signal = iPose,base64plot=base64)
# pose_by_selection
class PoseHandService(services_pb2_grpc.PoseHandService):
  def PoseHand_manual(self,request,context):
    classifier.pose_by_selection(requested_pose=request.pose)
    # classifier.pose_by_selection(requested_pose=0)
    print('grpc site, receive pose: %s',request.pose)
    return services_pb2.PoseHand()
      

def serve():
  DEFAULT_PORT = 50055
  # Get the port number from the command line parameter    
  port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
  HOST = f'localhost:{port}'

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  classifier.BlueToothConnect()
  print("YABAI!")

  # TODO, add your gRPC service to self-hosted server, e.g.
  services_pb2_grpc.add_NumberSortingServiceServicer_to_server(NumberSortingService(), server)
  services_pb2_grpc.add_UrMumJokeServicer_to_server(UrMumJoke(),server)
  services_pb2_grpc.add_EMGClassifierServiceServicer_to_server(EMGClassifierService(),server)
  services_pb2_grpc.add_PoseHandServiceServicer_to_server(PoseHandService(),server)
  health_pb2_grpc.add_HealthServicer_to_server(health.HealthServicer(), server)

  server.add_insecure_port(HOST)
  print(f"gRPC server started and listening on {HOST}")
  server.start()
  server.wait_for_termination()
  
if __name__ == '__main__':
  serve()  


# START OF NN Classifier part

