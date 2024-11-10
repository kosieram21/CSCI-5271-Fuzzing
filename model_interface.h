#include <unistd.h>

typedef struct {
    int writePipe[2];
    int readPip[2];

} ModelInterface;

int InitializeInterface(ModelInterface* interface) {

}