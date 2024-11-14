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

        self.readPipe = open(FIFO_C_TO_PY, 'rb')
        self.writePipe = open(FIFO_PY_TO_C, 'wb')

    def close(self):
        self.readPipe.close()
        self.writePipe.close()

    def receive_command(self):
        header = self.readPipe.read(4)
        payload_size = struct.unpack('<I', header)[0]
        payload = self.readPipe.read(payload_size)
        command, args = payload.decode('utf-8').split(':', 1)
        return command, args.encode('utf-8')

    def send_response(self, payload):
        payload_size = len(payload)
        print(payload_size)
        header = payload_size.to_bytes(4, byteorder='little')
        self.writePipe.write(header)
        self.writePipe.write(payload)

model_interface = ModelInterface()
model_interface.open()
processing = True

while processing:
    command, args = model_interface.receive_command()

    if command == 'Close':
        print('closing...')
        model_interface.send_response((0).to_bytes(1, byteorder='little'))
        model_interface.close()
        processing = False
    elif command == 'GetAction':
        print('getting action...')
        state_size = struct.unpack('<I', args[:4])[0]
        state = args[4:].decode('utf-8')
        print(state_size)
        print(state)
        try:
            error_code = 0
            action = 3  #number corresponding to model output of what action to take
        except:
            error_code = 1
            action = 0  #do nothing, we failed
        response_payload = (error_code).to_bytes(1, byteorder='little')
        response_payload += (action).to_bytes(1, byteorder='little')
        model_interface.send_response(response_payload)



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
