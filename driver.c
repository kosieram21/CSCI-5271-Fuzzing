#include "model_interface.h"

int main() {
    ModelInterface interface;
    if (OpenInterface(&interface)) {
        return -1;
    }

    if (CloseInterface(&interface)) {
        return -1;
    }
    
    return 0;
}