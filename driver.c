#include "model_interface.h"

int main() {
    ModelInterface interface;
    if (OpenInterface(&interface)) {
        return -1;
    }

    char* output;
    int outputSize;
    if (GenerateMutation(&interface, &output, &outputSize)) {
        return -1;
    }

    printf("%.*s\n", outputSize, output);

    int codeCoverage = 100
    if (UpdateModel(&interface, codeCoverage)) {
        return -1;
    }

    if (CloseInterface(&interace)) {
        return -1;
    }
    
    return 0;
}