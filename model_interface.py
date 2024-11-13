import os
import struct

FIFO_C_TO_PY = 'c_to_py_fifo'
FIFO_PY_TO_C = 'py_to_c_fifo'

class ModelInterface():
    def __init__(self):
        self.readPipe = None
        self.writePipe = None

    def open(self):
        if not os.path.exists(FIFO_C_TO_PY):
            os.mkfifo(FIFO_C_TO_PY)
        if not os.path.exists(FIFO_PY_TO_C):
            os.mkfifo(FIFO_PY_TO_C)

        self.readPipe = open(FIFO_C_TO_PY, 'r')
        self.writePipe = open(FIFO_PY_TO_C, 'w')

    def close(self):
        self.readPipe.close()
        self.writePipe.close()

    def receive_command(self):
        header = bytes(self.readPipe.read(4), encoding='utf-8')
        payload_size = struct.unpack('>I', header)[0]
        print(payload_size)
        payload = self.readPipe.read(payload_size)
        print(bytes(payload, encoding='utf-8'))
        command, args = payload.split(':', 1)
        # we need to be able to decode the arg string for other commands
        return command, args

    def send_response(self, payload):
        payload_size = len(payload)
        header = payload_size.to_bytes(4, byteorder='big')
        self.writePipe.write(header)
        self.writePipe.write(payload)

model_interface = ModelInterface()
model_interface.open()
processing = True

while processing:
    command, args = model_interface.receive_command()
    print(command)

    if command == 'Close':
        print('closing...')
        model_interface.send_response(str(0))
        model_interface.close()
        processing = False

    # if command == 'GenerateMutation':
    #     print('mutating...')
    #     output = "abcd1234"
    #     outputSize = str(len(output)).zfill(10)
    #     writePipe.write(outputSize)
    #     writePipe.flush()
    #     writePipe.write(output)
    #     writePipe.flush()
    #     writePipe.write('0')
    #     writePipe.flush()
    # elif command == 'UpdateModel':
    #     print('updating...')
    #     code_coverage = readPipe.readline()
    #     print(f'code coverage: {code_coverage}')
    #     writePipe.write('0')
    #     writePipe.flush()
    # elif command == 'Close':
    #     print('closing...')
    #     writePipe.write('0')
    #     writePipe.flush()
    #     readPipe.close()
    #     writePipe.close()
    #     processing = False
    # else:
    #     print(f'{command} is an invalid command')
    #     processing = False
