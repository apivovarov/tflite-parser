#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hexagon_nn.h>
#include <hexagon_nn_ops.h>
#include <sdk_fastrpc.h>

#include "dlr_hexagon_model.h"


int dlr_hexagon_model_init(int* graph_id_ptr, uint8_t** input, uint8_t** output, int debug_level) {
    int err = 0;

    if ((err = fastrpc_setup()) != 0) return err;

    hexagon_nn_config();

    if ((err = hexagon_nn_init(graph_id_ptr)) != 0) {
        printf("hexagon_nn_init failed: %d\n", err);
        return err;
    }

    if ((err = hexagon_nn_set_powersave_level(0)) != 0) {
        printf("hexagon_nn_set_powersave_level failed: %d\n", err);
        return err;
    }

    int graph_id = *graph_id_ptr;

    hexagon_nn_set_debug_level(graph_id, debug_level);

    dlr_hexagon_init_graph(graph_id);

    if ((err = hexagon_nn_prepare(graph_id)) != 0) {
        printf("hexagon_nn_prepare failed: %d\n", err);
        return err;
    }

    if ((*output = rpcmem_alloc(
             ION_HEAP_ID_SYSTEM,
             RPCMEM_DEFAULT_FLAGS,
             OUT_SIZE)
            ) == NULL) {
        printf("rpcmem_alloc failed for output. OUT_SIZE=%d bytes\n", OUT_SIZE);
        err=1;
        return err;
    }

    if ((*input = rpcmem_alloc(
             ION_HEAP_ID_SYSTEM,
             RPCMEM_DEFAULT_FLAGS,
             IN_SIZE)
            ) == NULL) {
        printf("rpcmem_alloc failed for input. IN_SIZE=%d bytes\n", IN_SIZE);
        err=1;
        return err;
    }

    return 0;
}

int dlr_hexagon_model_exec(int graph_id, uint8_t* input, uint8_t* output) {
    int err = 0;
    uint32_t out_batches, out_height, out_width, out_depth, out_data_size;
    err = hexagon_nn_execute(
             graph_id,
             IN_BATCH,
             IN_HEIGHT,
             IN_WIDTH,
             IN_DEPTH,
             (const uint8_t *)input, // Pointer to input data
             IN_SIZE,                // How many total bytes of input?
             (unsigned int*) &out_batches,
             (unsigned int*) &out_height,
             (unsigned int*) &out_width,
             (unsigned int*) &out_depth,
             (uint8_t *)output,      // Pointer to output buffer
             OUT_SIZE,               // Max size of output buffer
             (unsigned int*) &out_data_size);         // Actual size used for output

    if (err != 0) {
        printf("hexagon_nn_execute failed: %d\n", err);
        return err;
    }

    // Sanity check that our output is sized as expected,
    //   else we might have built our graph wrong.
    if ( (out_batches != OUT_BATCH) ||
         (out_height != OUT_HEIGHT) ||
         (out_width != OUT_WIDTH) ||
         (out_depth != OUT_DEPTH) ||
         (out_data_size != OUT_SIZE) ) {
            printf("Output sizing seems wrong: (%ux%ux%ux%u %u)\n",
                   (unsigned int) out_batches, (unsigned int) out_height,
                   (unsigned int) out_width, (unsigned int) out_depth,
                   (unsigned int) out_data_size
                    );
            err = 1;
            return err;
    }
    return 0;
}

void dlr_hexagon_model_close(int graph_id) {
    // Free the memory, especially if we want to build subsequent graphs
    hexagon_nn_teardown(graph_id);

    // Stop fastRPC
    fastrpc_teardown();
}

int dlr_hexagon_nn_getlog(int graph_id, unsigned char* buff, int sz) {
  return hexagon_nn_getlog(graph_id, buff, sz);
}
