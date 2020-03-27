#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

int dlr_hexagon_model_init(int* graph_id_ptr, uint8_t** input, uint8_t** output, int debug_level);
int dlr_hexagon_model_exec(int graph_id, uint8_t* input, uint8_t* output);
void dlr_hexagon_model_close(int graph_id);
int hexagon_nn_getlog(int graph_id, unsigned char *buf, uint32_t length);

long long ns();
size_t read_img(const char *path, uint8_t *img, size_t sz);

int main() {
    int graph_id = 0;
    uint8_t* input = NULL;
    uint8_t* output = NULL;
    int debug_level = 1;
    int err = 0;
    int buff_size = 1024*1024;
    char * buff = (char *) malloc(buff_size+1);
    err = dlr_hexagon_model_init(&graph_id, &input, &output, debug_level);
    if (err != 0) {
      printf("dlr_hexagon_model_init failed %d\n", err);
      return err;
    }

    hexagon_nn_getlog(graph_id, (unsigned char *) buff, buff_size);
    printf("hexagon_nn_getlog (%zu):\n%s\n", strlen((const char *) buff), buff);

    read_img("/data/local/tmp/vendor/bin/cat224-3.txt", input, 224*224*3);

    err = dlr_hexagon_model_exec(graph_id, input, output);
    if (err != 0) {
      printf("dlr_hexagon_model_exec failed %d\n", err);
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

    //hexagon_nn_getlog(graph_id, (unsigned char *) buff, buff_size);
    //printf("hexagon_nn_getlog (%zu):\n%s\n", strlen((const char *) buff), buff);

    dlr_hexagon_model_close(graph_id);
    input = NULL;
    output = NULL;
    graph_id = 0;
    return 0;
}

long long ns() {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000000)+(tv.tv_usec);
}

size_t read_img(const char *path, uint8_t *img, size_t sz) {
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
