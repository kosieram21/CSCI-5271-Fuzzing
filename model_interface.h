#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#define FIFO_C_TO_PY "c_to_py_fifo"
#define FIFO_PY_TO_C "py_to_c_fifo"

typedef struct {
    int writePipe;
    int readPipe;
} ModelInterface;

int SendCommand(const ModelInterface* const interface, const unsigned char* const payload, const unsigned int payloadSize) 
{
    if (interface == NULL) {
        printf("SendCommand: interface must not be NULL\n");
        return -1;
    }

    unsigned char header[4];
    memcpy(header, &payloadSize, 4);

    if (write(interface->writePipe, header, sizeof(header)) != sizeof(header)) {
        printf("SendCommand: failed to write header\n");
        return -1;
    }

    if (write(interface->writePipe, payload, payloadSize) != payloadSize) {
        printf("SendCommand: failed to write payload\n");
        return -1;
    }

    return 0;
}

int ReceiveResponse(const ModelInterface* const interface, unsigned char** payload, unsigned int* payloadSize) {
    if (interface == NULL) {
        printf("ReceiveResponse: interface must not be NULL\n");
        return -1;
    }

    unsigned char header[4];
    if (read(interface->readPipe, header, sizeof(header)) != sizeof(header)) {
        printf("ReceiveResponse: failed to read header\n");
        return -1;
    }

    memcpy(payloadSize, header, 4);
    *payload = (unsigned char*)malloc(*payloadSize);

    if (read(interface->readPipe, *payload, *payloadSize) != *payloadSize) {
        free(payload);
        printf("ReceiveResponse: failed to read payload\n");
        return -1;
    }

    return 0;
}

int OpenInterface(ModelInterface* const interface) {
    if (interface == NULL) {
        printf("OpenInterface: interface must not be NULL\n");
        return -1;
    }

    if (mkfifo(FIFO_C_TO_PY, 0666) && errno != EEXIST) {
        printf("OpenInterface: failed to make write pipe\n");
        return -1;
    }

    if (mkfifo(FIFO_PY_TO_C, 0666) && errno != EEXIST) {
        printf("OpenInterface: failed to make read pipe\n");
        return -1;
    }

    interface->writePipe = open(FIFO_C_TO_PY, O_WRONLY);
    if (interface->writePipe == -1) {
        printf("OpenInterface: failed to open write pipe\n");
        return -1;
    }

    interface->readPipe = open(FIFO_PY_TO_C, O_RDONLY);
    if (interface->readPipe == -1) {
        printf("OpenInterface: failed to open read pipe\n");
    }

    return 0;
}

int CloseInterface(ModelInterface* const interface) {
    if (interface == NULL) {
        printf("CloseInterface: interface must not be NULL\n");
        return -1;
    }

    if (SendCommand(interface, "Close:", 6)) {
        printf("CloseInterface: failed to send command\n");
        return -1;
    }

    unsigned char* responsePayload;
    unsigned int responseSize;
    if (ReceiveResponse(interface, &responsePayload, &responseSize) || responsePayload == NULL) {
        printf("CloseInterface: failed to recieve response\n");
        return -1;
    }

    if (responseSize != 1 || responsePayload[0]) {
        free(responsePayload);
        printf("CloseInterface: command execution failed\n");
        return -1;
    }

    free(responsePayload);

    if (close(interface->writePipe)) {
        printf("CloseInterface: failed to close write pipe\n");
        return -1;
    }

    if (close(interface->readPipe)) {
        printf("CloseInterface: failed to close read pipe\n");
        return -1;
    }

    return 0;
}

// Give the model a state (program input) it will choose an action (input mutation)
int GetAction(const ModelInterface* const interface, 
    const char* const state, const unsigned int stateSize, 
    unsigned char* action) {
    if (interface == NULL) {
        printf("GetAction: interface must not be NULL\n");
        return -1;
    }

    const unsigned int commandPayloadSize = 14 + stateSize;
    unsigned char* commandPayload = (unsigned char*)malloc(commandPayloadSize);
    memcpy(commandPayload, "GetAction:", 10);
    memcpy(commandPayload + 10, &stateSize, 4);
    memcpy(commandPayload + 14, state, stateSize);

    if (SendCommand(interface, commandPayload, commandPayloadSize)) {
        free(commandPayload);
        printf("GetAction: failed to send command\n");
        return -1;
    }

    free(commandPayload);

    unsigned char* responsePayload;
    unsigned int responseSize;
    if (ReceiveResponse(interface, &responsePayload, &responseSize) || responsePayload == NULL) {
        printf("GetAction: failed to recieve response\n");
        return -1;
    }

    if (responseSize != 2 || responsePayload[0]) {
        free(responsePayload);
        printf("GetAction: command execution failed\n");
        return -1;
    }

    *action = responsePayload[1];

    free(responsePayload);

    return 0;
}

// Recording an experience for training the model. We are going to use the bellman equation as the objective function
// Q(s,a) = r(s,a) + max_a' Q(s', a') => regress on (Q(s,a) - (r(s,a) + max_a' Q(s', a')))^2 (squared error)
int RecordExperience(const ModelInterface* const interface,
    const char* const state, const unsigned int stateSize,
    const char* const nextState, const unsigned int nextStateSize,
    const unsigned char action, const unsigned int reward) {
    if (interface == NULL) {
        printf("RecordExperience: interface must not be NULL\n");
        return -1;
    }

    const unsigned int commandPayloadSize = 30 + stateSize + nextStateSize;
    unsigned char* commandPayload = (unsigned char*)malloc(commandPayloadSize);
    memcpy(commandPayload, "RecordExperience:", 17);
    memcpy(commandPayload + 17, &stateSize, 4);
    memcpy(commandPayload + 21, state, stateSize);
    memcpy(commandPayload + stateSize + 21, &nextStateSize, 4);
    memcpy(commandPayload + stateSize + 25, nextState, nextStateSize);
    memcpy(commandPayload + stateSize + nextStateSize + 25, &action, 1);
    memcpy(commandPayload + stateSize + nextStateSize + 26, &reward, 4);

    if (SendCommand(interface, commandPayload, commandPayloadSize)) {
        free(commandPayload);
        printf("RecordExperience: failed to send command\n");
        return -1;
    }

    free(commandPayload);

    unsigned char* responsePayload;
    unsigned int responseSize;
    if (ReceiveResponse(interface, &responsePayload, &responseSize) || responsePayload == NULL) {
        printf("RecordExperience: failed to recieve response\n");
        return -1;
    }

    if (responseSize != 1 || responsePayload[0]) {
        free(responsePayload);
        printf("RecordExperience: command execution failed\n");
        return -1;
    }

    free(responsePayload);

    return 0;
}

// Replay all the experiences from the episode to update the model weights
int ReplayExperiences(const ModelInterface* const interface) {
    if (interface == NULL) {
        printf("ReplayExperiences: interface must not be NULL\n");
        return -1;
    }

    if (SendCommand(interface, "ReplayExperiences:", 18)) {
        printf("ReplayExperiences: failed to send command\n");
        return -1;
    }

    unsigned char* responsePayload;
    unsigned int responseSize;
    if (ReceiveResponse(interface, &responsePayload, &responseSize) || responsePayload == NULL) {
        printf("ReplayExperiences: failed to recieve response\n");
        return -1;
    }

    if (responseSize != 1 || responsePayload[0]) {
        free(responsePayload);
        printf("ReplayExperiences: command execution failed\n");
        return -1;
    }

    free(responsePayload);

    return 0;
}