import os

FIFO_C_TO_PY = 'c_to_py_fifo'
FIFO_PY_TO_C = 'py_to_c_fifo'

if not os.path.exists(FIFO_C_TO_PY):
    os.mkfifo(FIFO_C_TO_PY)
if not os.path.exists(FIFO_PY_TO_C):
    os.mkfifo(FIFO_PY_TO_C)

readPipe = open(FIFO_C_TO_PY, 'r')
writePipe = open(FIFO_PY_TO_C, 'w')
processing = True

while processing:
    command = readPipe.readline().strip()

    if command == 'GenerateMutation':
        print('mutating...')
        output = "abcd1234"
        outputSize = str(len(output)).zfill(10)
        writePipe.write(outputSize)
        writePipe.flush()
        writePipe.write(output)
        writePipe.flush()
        writePipe.write('0')
        writePipe.flush()
    elif command == 'UpdateModel':
        print('updating...')
        code_coverage = readPipe.readline()
        print(f'code coverage: {code_coverage}')
        writePipe.write('0')
        writePipe.flush()
    elif command == 'Close':
        print('closing...')
        writePipe.write('0')
        writePipe.flush()
        readPipe.close()
        writePipe.close()
        processing = False
    else:
        print(f'{command} is an invalid command')
        processing = False
