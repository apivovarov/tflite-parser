#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "qiv2.h"

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

int main(int argc, char **argv) {
        int err;
        time_t t;
        srand((unsigned) time(&t));

        if (fastrpc_setup() != 0) return 1;

        hexagon_nn_config();

        hexagon_nn_nn_id nn_id;
        if ((err = hexagon_nn_init(&nn_id)) != 0) {
                printf("Whoops... Cannot init: %d\n", err);
                goto TEARDOWN;
        }

        // Set power level (to max/turbo)
        if ((err = hexagon_nn_set_powersave_level(0)) != 0) {
                printf("Whoops... Cannot set power level: %d\n", err);
                goto TEARDOWN;
        }

        err |= hexagon_nn_set_debug_level(nn_id, 0);
        ERRCHECK

        init_graph(nn_id);

        if ((err = hexagon_nn_prepare(nn_id)) != 0) {
                printf("Whoops... Cannot prepare: %d\n", err);
                goto TEARDOWN;
        }

        uint8_t *output;
        if ((output = rpcmem_alloc(
                     ION_HEAP_ID_SYSTEM,
                     RPCMEM_DEFAULT_FLAGS,
                     OUT_SIZE)
                    ) == NULL) {
                printf("Whoops... Cannot malloc for outputs\n");
                err=1;
                goto TEARDOWN;
        }

        uint8_t *input;
        if ((input = rpcmem_alloc(
                     ION_HEAP_ID_SYSTEM,
                     RPCMEM_DEFAULT_FLAGS,
                     IN_SIZE)
                    ) == NULL) {
                printf("Whoops... Cannot malloc for inputs\n");
                err=1;
                goto TEARDOWN;
        }

        //read_img("/data/local/tmp/vendor/bin/cat224-3.txt", input, IN_LEN);
        read_img("/data/local/tmp/vendor/bin/cat224-3.txt", input, IN_LEN);
    int N = 5;
    float totalT = 0.0;
    uint8_t* out_8 = output;
    float* out_f = (float*) output;
    for (int n = -1; n < N; n++) {
        long long t1 = ns();
        unsigned int out_batches, out_height, out_width, out_depth, out_data_size;

        err = hexagon_nn_execute(
                     nn_id,
                     IN_BATCH,
                     IN_HEIGHT,
                     IN_WIDTH,
                     IN_DEPTH,
                     (const uint8_t *)input, // Pointer to input data
                     IN_SIZE,                // How many total bytes of input?
                     &out_batches,
                     &out_height,
                     &out_width,
                     &out_depth,
                     output,      // Pointer to output buffer
                     OUT_SIZE,               // Max size of output buffer
                     &out_data_size);         // Actual size used for output

        if (n == -5) {
          int buff_size = 1024*1024;
          char * buff = (char *) malloc(buff_size+1);
          hexagon_nn_getlog(nn_id, (unsigned char *) buff, buff_size);
          printf("hexagon_nn_getlog (%zu):\n%s\n", strlen((const char *) buff), buff);
        }
        if (err != 0) {
            printf("Whoops... run failed: %d\n",err);
            goto TEARDOWN;
        }
        // Sanity check that our output is sized as expected,
        //   else we might have built our graph wrong.
        if ( (out_batches != OUT_BATCH) ||
             (out_height != OUT_HEIGHT) ||
             (out_width != OUT_WIDTH) ||
             (out_depth != OUT_DEPTH) ||
             (out_data_size != OUT_SIZE) ) {
                printf("Whoops... Output sizing seems wrong: (%ux%ux%ux%u %u)\n",
                       out_batches, out_height, out_width, out_depth,
                       out_data_size
                        );
                goto TEARDOWN;
        }
        long long t2 = ns();
        float tt = (t2-t1)/1000.0;
        if (n > -1) totalT += tt;
        if (OUT_ELEMENTSIZE == 1) {
          printf("%d: OUTPUT: %d:%d, time: %.3f\n", n, 282, out_8[282], tt);
        } else {
          printf("%d: OUTPUT: %d:%.8f, time: %.3f\n", n, 282, out_f[282], tt);
        }
    }
    printf("AVG Time: %.3f\n", totalT/N);
        /*int rid = 0;
        int cid = 0;
        float min = 2000000000;
        float max = -2000000000;
        for (int i = 0; i < 112*112*24; i++) {
          if (i%24 == 0) {
            if (output[i] > max) max = output[i];
            if (output[i] < min) min = output[i];
            cid++;
            if (cid == 112) {
              printf("%d : %.8f %8f\n", rid, min, max);
              rid++;
              cid = 0;
              min = 2000000000;
              max = -2000000000;
            }
          }
        }*/

        printf("Got input:  [ ");
        for (int i = 0; i < 4; i++) {
          printf("%3d ", input[i]);
        }
        printf("]\n");
        printf("Got output:  [ ");
        for (int i = 0; i < 10; i++) {
          if (OUT_ELEMENTSIZE == 1) {
            printf("%d , ", out_8[i]);
          } else {
            printf("%.8f , ", out_f[i]);
          }
        }
        printf("]\n");
        /*int out_len = 112*112*24;
        printf("Got output:  [ ");
        for (int i = out_len-10; i < out_len; i++) {
          printf("%.8f ", output[i]);
        }
        printf("]\n");*/
       float mx = 0;
       int mx_id = 0;
       printf("Top outputs:  [ ");
       for(int i=0; i < OUT_LEN; i++) {
         if (OUT_ELEMENTSIZE == 1) {
           if (out_8[i] > mx) {
             mx = out_8[i];
             mx_id = i;
           }
           if (out_8[i] > 0.42*255) {
             //printf("%d:%d, ", i, out_8[i]);
           }
         } else {
           if (out_f[i] > 0.2) {
             //printf("%d:%.8f, ", i, out_f[i]);
           }
         }
       }
       printf("\n");
       printf("Max: [%d]=%.3f\n", mx_id, mx);

TEARDOWN:
        // Free the memory, especially if we want to build subsequent graphs
        hexagon_nn_teardown(nn_id);

        // Stop fastRPC
        fastrpc_teardown();

        return err;
}
