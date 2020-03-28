void info_for_debug(unsigned int id, const char *name, const char *opname) {
  printf("Appending %x, %s, op %s\n", id, name, opname);
}

#define OUTPUT_4D(B,H,W,D,ES) { .rank = 4, .max_sizes = {B,H,W,D}, .elementsize=ES, }
#define ALIGNED __attribute__((aligned(128)))

#define APPEND_CONST_NODE(ID,...) if (hexagon_nn_append_const_node(nn_id,ID,__VA_ARGS__) != 0) \
        printf("node %d returned nonzero\n",ID)
#define APPEND_NODE(NAME,ID,OP,...) info_for_debug(ID,NAME,#OP); \
        if (hexagon_nn_append_node(nn_id,ID,OP,__VA_ARGS__) != 0) \
            printf("node %d <%s/%s> returned nonzero\n",ID,NAME,#OP)


