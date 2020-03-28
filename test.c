#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

int main() {
    void *handle;
    double (*cosine)(double);
    char *error;

    handle = dlopen ("/usr/lib/x86_64-linux-gnu/libm.so", RTLD_LAZY);
    if (!handle) {
        fputs(dlerror(), stderr);
        exit(1);
    }

    cosine = dlsym(handle, "cos");
    if (cosine != NULL)  {
        fputs(dlerror(), stderr);
        exit(1);
    }

    printf ("%f\n", (*cosine)(2.0));
    dlclose(handle);
}
