import os

FIFO_C_TO_PY = 'c_to_py_fifo'
FIFO_PY_TO_C = 'py_to_c_fifo'

readPipe = open(FIFO_C_TO_PY, 'r')
writePipe = open(FIFO_PY_TO_C, 'w')
processing = True

while processing:
    command = readPipe.readline()

    if command == 'GenerateMutation':
        print('mutating...')
        output = "abcd1234"
        outputSize = str(len(output)).zfill(10)
        writePipe.write(outputSize)
        writePipe.write(output)
    elif command == 'UpdateModel':
        print('updating...')
        code_coverage = readPipe.readline()
    elif command == 'Close':
        readPipe.close()
        writePipe.close()
        processing = False
    else:
        print('Invalid Command')
