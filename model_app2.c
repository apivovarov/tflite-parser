#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#define PRINT_BUFSIZE (2*1024*1024)

int (*dlr_hexagon_model_init)(int*, uint8_t**, uint8_t**, int);
int (*dlr_hexagon_model_exec)(int, uint8_t*, uint8_t*);
void (*dlr_hexagon_model_close)(int);
int (*dlr_hexagon_nn_getlog)(int, unsigned char*, int);

static long long ns();
static size_t read_img(const char *path, uint8_t *img, size_t sz);
static void* dlr_find_symbol(void* handle, const char* fn_name);
static void print_hexagon_nn_log(int graph_id, char* buff);


int main() {
    const char* model = "/data/local/tmp/vendor/lib64/dlr_hexagon_model.so";
    void* handle  = dlopen (model, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        printf("dlopen error: %s\n", dlerror());
        exit(1);
    }

    dlr_hexagon_model_init = dlr_find_symbol(handle, "dlr_hexagon_model_init");
    dlr_hexagon_model_exec = dlr_find_symbol(handle, "dlr_hexagon_model_exec");
    dlr_hexagon_model_close = dlr_find_symbol(handle, "dlr_hexagon_model_close");
    dlr_hexagon_nn_getlog = dlr_find_symbol(handle, "hexagon_nn_getlog");
    int graph_id = 0;
    uint8_t* input = NULL;
    uint8_t* output = NULL;
    int debug_level = 0;
    int err = 0;
    char* buff = (char*) malloc(PRINT_BUFSIZE);
    if (buff == NULL) {
      printf("Can not allocate print buffer, sizee: %d\n", PRINT_BUFSIZE);
      return -1;
    }
    err = (*dlr_hexagon_model_init)(&graph_id, &input, &output, debug_level);
    if (err != 0) {
      printf("dlr_hexagon_model_init failed %d\n", err);
      print_hexagon_nn_log(graph_id, buff);
      return err;
    }
    print_hexagon_nn_log(graph_id, buff);

    read_img("/data/local/tmp/vendor/bin/cat224-3.txt", input, 224*224*3);

    err = (*dlr_hexagon_model_exec)(graph_id, input, output);
    if (err != 0) {
      printf("dlr_hexagon_model_exec failed %d\n", err);
      print_hexagon_nn_log(graph_id, buff);
      return err;
    }
    float mx = 0;
    int mx_id = 0;
    for(int i=0; i < 1001; i++) {
        if (output[i] > mx) {
            mx = output[i];
            mx_id = i;
        }
    }
    printf("Max: [%d]=%.3f\n", mx_id, mx);

    (*dlr_hexagon_model_close)(graph_id);
    input = NULL;
    output = NULL;
    graph_id = 0;
    return 0;
}

static void* dlr_find_symbol(void* handle, const char* fn_name) {
    printf("Loading %s\n", fn_name);

    void* fn = dlsym(handle, fn_name);
    if (!fn)  {
        printf("dlsym error for %s: %s\n", fn_name, dlerror());
        exit(1);
    }
    return fn;
}

static long long ns() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000000)+(tv.tv_usec);
}

static size_t read_img(const char *path, uint8_t *img, size_t sz) {
    printf("img_name: %s\n", path);

    FILE * fp;
    fp = fopen(path, "r");
    if (fp == NULL) {
      printf("Dummy image\n");
      for (int i = 0; i < sz; i++) {
        img[i] = 10+(rand() % 241);
      }
      return sz;
    } else {
      int max_len = 80;
      char line[max_len+1];
      int i = 0;
      while (fgets(line, max_len, fp) != NULL && i < sz) {
        uint8_t v = (uint8_t) atoi(line);
        img[i] = v;
        i++;
      }

      fclose(fp);

      printf("Image read ok, size: %d\n", i);
      return i;
    }
}

static void print_hexagon_nn_log(int graph_id, char* buff) {
    int err = (*dlr_hexagon_nn_getlog)(graph_id, (unsigned char *) buff, PRINT_BUFSIZE);
    if (err == 0) {
      puts(buff);
    }
}
