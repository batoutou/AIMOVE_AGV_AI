import zmq

context = zmq.Context()

print("Connecting to hello world server…")  #  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://172.20.10.2:5555")
# socket.connect("tcp://172.20.10.14:5555")
message_robot="Connection OK"

print("Sending request : {}".format(message_robot))
socket.send(message_robot.encode())

message = socket.recv()#  Get the reply.
print("Received reply %s" % (message))
