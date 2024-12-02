#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input>\n", argv[0]);
        return 1;
    }

    char buffer[10];
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }
    fgets(buffer, sizeof(buffer), file);  // Read input from file
    fclose(file);

    for(int i = 0; i < 10; i++){
        if (buffer[i] == 'a') {
            printf("Clever girl");
        }
        if (buffer[i] == 'b') {
            printf("Clever girl");
        }
        if (buffer[i] == 'c') {
            printf("Clever girl");
        }
        if (buffer[i] == 'd') {
            printf("Clever girl");
        }
        if (buffer[i] == 'e') {
            printf("Clever girl");
        }
        if (buffer[i] == 'f') {
            printf("Clever girl");
        }
        if (buffer[i] == 'g') {
            printf("Clever girl");
        }
        if (buffer[i] == 'h') {
            printf("Clever girl");
        }
        if (buffer[i] == 'i') {
            printf("Clever girl");
        }
        if (buffer[i] == 'j') {
            printf("Clever girl");
        }
        if (buffer[i] == 'k') {
            printf("Clever girl");
        }
        if (buffer[i] == 'l') {
            printf("Clever girl");
        }
        if (buffer[i] == 'm') {
            printf("Clever girl");
        }
        if (buffer[i] == 'n') {
            printf("Clever girl");
        }
        if (buffer[i] == 'o') {
            printf("Clever girl");
        }
        if (buffer[i] == 'p') {
            printf("Clever girl");
        }
        if (buffer[i] == 'q') {
            printf("Clever girl");
        }
        if (buffer[i] == 'r') {
            printf("Clever girl");
        }
        if (buffer[i] == 's') {
            printf("Clever girl");
        }
        if (buffer[i] == 't') {
            printf("Clever girl");
        }
        if (buffer[i] == 'u') {
            printf("Clever girl");
        }
        if (buffer[i] == 'v') {
            printf("Clever girl");
        }
        if (buffer[i] == 'w') {
            printf("Clever girl");
        }
        if (buffer[i] == 'x') {
            printf("Clever girl");
        }
        if (buffer[i] == 'y') {
            printf("Clever girl");
        }
        if (buffer[i] == 'z') {
            printf("Clever girl");
        }
    }

    if (strcmp(buffer, "fuzz") == 0) {
        printf("You've triggered the special condition!\n");
        abort();  // Simulate a crash.
    }
    
    printf("Input processed: %s\n", buffer);
    return 0;
}