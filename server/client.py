from __future__ import print_function

import logging

import grpc
from grpc_generated import services_pb2_grpc,services_pb2


DEFAULT_PORT = 50055

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    url = "localhost:"+str(DEFAULT_PORT)
    print("client start at: %s"%url)

    with grpc.insecure_channel(url) as channel:
        stub = services_pb2_grpc.UrMumJokeStub(channel)
        response = stub.TellJoke(services_pb2.theJokeReq(punchline="can not breath without dying",reason="so fat"))
        stub_classifier  = services_pb2_grpc.EMGClassifierServiceStub(channel)
        
    print("client received: " + response.message)


if __name__ == "__main__":
    logging.basicConfig()
    run()
