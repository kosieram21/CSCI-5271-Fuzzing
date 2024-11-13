#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#define FIFO_C_TO_PY "c_to_py_fifo"
#define FIFO_PY_TO_C "py_to_c_fifo"

typedef struct {
    int writePipe;
    int readPipe;
} ModelInterface;

int SendCommand(const ModelInterface* const interface, const char* const payload, const size_t payloadSize) 
{
    if (interface == NULL) {
        return -1;
    }

    const unsigned char header[4] = { payloadSize >> 24, payloadSize >> 16, payloadSize >> 8, payloadSize };

    if (write(interface->writePipe, header, sizeof(header))) {
        return -1;
    }

    printf("%s", payload);

    if (write(interface->writePipe, payload, payloadSize)) {
        return -1;
    }

    return 0;
}

int ReceiveResponse(const ModelInterface* const interface, char** payload, size_t* payloadSize) {
    if (interface == NULL) {
        return -1;
    }

    unsigned char header[4];
    if (read(interface->readPipe, header, sizeof(header))) {
        return -1;
    }

    *payloadSize = header[3] << 24 | header[2] << 16 | header[1] << 8 | header[0];

    if (read(interface->readPipe, &payload, *payloadSize)) {
        return -1;
    }

    return 0;
}

int OpenInterface(ModelInterface* const interface) {
    if (interface == NULL) {
        return -1;
    }

    if (mkfifo(FIFO_C_TO_PY, 0666) && errno != EEXIST) {
        return -1;
    }

    if (mkfifo(FIFO_PY_TO_C, 0666) && errno != EEXIST) {
        return -1;
    }

    interface->writePipe = open(FIFO_C_TO_PY, O_WRONLY);
    interface->readPipe = open(FIFO_PY_TO_C, O_RDONLY);

    return 0;
}

int CloseInterface(ModelInterface* const interface) {
    if (interface == NULL) {
        return -1;
    }

    if (SendCommand(interface, "Close:", 6)) {
        return -1;
    }

    char* responsePayload;
    size_t responseSize;
    if (ReceiveResponse(interface, &responsePayload, &responseSize)) {
        return -1;
    }

    if (responseSize != 1 || responsePayload[0]) {
        return -1;
    }

    if (close(interface->writePipe)) {
        return -1;
    }

    if (close(interface->readPipe)) {
        return -1;
    }

    return 0;
}

int GetAction(const ModelInterface* const interface, 
    const char* const state, const unsigned int stateSize, 
    unsigned int* action) {
    // Give the model a state (program input) it will choose an action (input mutation)

    return 0;
}

int RecordExperience(const ModelInterface* const interface,
    const char* const state, const unsigned int stateSize,
    const char* const nextState, const unsigned int nextStateSize,
    const unsigned int action, const unsigned int reward) {
    // Recording an experience for training the model. We are going to use the bellman equation as the objective function
    // Q(s,a) = r(s,a) + max_a' Q(s', a') => regress on (Q(s,a) - (r(s,a) + max_a' Q(s', a')))^2 (squared error)

    return 0;
}

int ReplayExperiences(const ModelInterface* const interface) {
    // Replay all the experiences from the episode to update the model weights

    return 0;
}

/*int GenerateMutation(const ModelInterface* const interface, char** output, int* outputSize) {
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
    
    char buffer[11];
    snprintf(buffer, sizeof(buffer), "%d\n", codeCoverage);
    write(interface->writePipe, buffer, sizeof(buffer));

    char errorBuffer;
    read(interface->readPipe, &errorBuffer, 1);

    return atoi(&errorBuffer);
}*/