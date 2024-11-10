#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#define FIFO_C_TO_PY "c_to_py_fifo"
#define FIFO_PY_TO_C "py_to_c_fifo"

typedef struct {
    int writePipe;
    int readPipe;

} ModelInterface;

int OpenInterface(ModelInterface* interface) {
    if (interface == NULL) {
        return -1;
    }

    mkfifo(FIFO_C_TO_PY, 0666);
    mkfifo(FIFO_PY_TO_C, 0666);

    interface->writePipe = open(FIFO_C_TO_PY, O_WRONLY);
    interface->readPipe = open(FIFO_PY_TO_C, O_RDONLY);

    return 0;
}

int CloseInterface(ModelInterface* interface) {
    if (interface == NULL) {
        return -1;
    }

    write(interface->writePipe, "Close\n", 6);

    char errorBuffer;
    read(interface->readPipe, &errorBuffer, 1);
    int error = atoi(&errorBuffer);
    if (error) {
        return error;
    }

    close(interface->writePipe);
    close(interface->readPipe);

    return 0;
}

int GenerateMutation(ModelInterface* interface, char** output, int* outputSize) {
    if (interface == NULL) {
        return -1;
    }

    write(interface->writePipe, "GenerateMutation\n", 17);
    
    char buffer[10];
    read(interface->readPipe, buffer, sizeof(buffer));

    int tmp = atoi(buffer);
    outputSize = &tmp;
    read(interface->readPipe, *output, *outputSize);

    char errorBuffer;
    read(interface->readPipe, &errorBuffer, 1);

    return atoi(&errorBuffer);
}

int UpdateModel(ModelInterface* interface, int codeCoverage) {
    if (interface == NULL) {
        return -1;
    }

    write(interface->writePipe, "UpdateModel\n", 12);
    
    char buffer[10];
    snprintf(buffer, sizeof(buffer), "%d", codeCoverage);
    write(interface->writePipe, buffer, sizeof(buffer));

    char errorBuffer;
    read(interface->readPipe, &errorBuffer, 1);

    return atoi(&errorBuffer);
}