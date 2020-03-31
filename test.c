#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

#define IN_NAME "Hello"

void getName(char** name, int** shape_ptr, int* dim) {
  *name = strcpy((char*) malloc(strlen(IN_NAME) + 1), IN_NAME);

  int* shape = (int*) malloc(4*sizeof(int));
  printf("Ptr %p\n", shape);
  shape[0] = 1;
  shape[1] = 224;
  shape[2] = 224;
  shape[3] = 3;
  *shape_ptr = shape;
  *dim = 4;
}

int main() {
    void *handle;
    double (*cosine)(double);
    char *error;

//    handle = dlopen ("/usr/lib/x86_64-linux-gnu/libm.so", RTLD_LAZY);
//    if (!handle) {
//        fputs(dlerror(), stderr);
//        exit(1);
//    }
//
//    cosine = dlsym(handle, "cos");
//    if (cosine != NULL)  {
//        fputs(dlerror(), stderr);
//        exit(1);
//    }
//
//    printf ("%f\n", (*cosine)(2.0));
//    dlclose(handle);

    //char* nn = NULL;
    char* name = NULL;
    int* shape = NULL;
    int dim;
    getName(&name, &shape, &dim);
    printf("Ptr %p\n", shape);

    printf("Name: %s\n", name);
    printf("Dim: %d, Shape: %d,%d,%d,%d\n", dim, shape[0], shape[1], shape[2], shape[3]);
}
