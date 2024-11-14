#include "model_interface.h"

int main() {
    ModelInterface interface;
    if (OpenInterface(&interface)) {
        return -1;
    }

    unsigned char action;
    if (GetAction(&interface, "test state", 10, &action)) {
        return -1;
    }

    printf("action: %d\n", action);

    if (RecordExperience(&interface, "test state", 10, "test state2", 11, action, 100)) {
        return -1;
    }

    if (CloseInterface(&interface)) {
        return -1;
    }
    
    return 0;
}