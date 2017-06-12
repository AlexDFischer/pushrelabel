#include "project.h"

#define ARRAYLEN(array) (sizeof(array) / sizeof(array[0]))

#define SOURCE_HEIGHT(graph) (graph->numVoxels + 2)
#define SINK_HEIGHT 0

struct vertex
{
  unsigned int height[2];
  unsigned int excess;
  unsigned int tempExcess[6];
  /**
   * capacities and flows are indexed the following way:
   *  - [0] is from source to voxel
   *  - [1] is from voxel to sink
   *  - [2] is from voxel to its neighbor with x-coordinate incremented
   *  - [3] is from voxel to its neighbor with y-coordinate incremented
   *  - [4] is from voxel to its neighbor with z-coordinate incremented
   */
  int capacities[5];
  int flows[5];
};

typedef struct vertex Vertex;

struct graph
{
  int width, height, depth;
  int numVoxels;
  Vertex *voxels;
};

typedef struct graph Graph;

void cudaMallocGraph(Graph **graph, int width, int height, int depth);

__device__ __host__
Vertex *getVertex(Graph *graph, int x, int y, int z);

__global__
void setCapacities(Graph *graph, Volume *volume, unsigned long minIntensity, unsigned long maxIntensity);

__global__
void storeSegmentation(Graph *graph, Volume *volume);

void pushRelabel(Graph *graph, Volume *volume);

void printGraph(Graph *graph);

int isMaxFlow(Graph *graph);

int pathfinderToSink(Graph *graph, int x, int y, int z, char visited[]);
