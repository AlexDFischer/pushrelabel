#include "project.h"
#include "volume.h"
#include "graph.h"

int getIndex(Graph *graph, int x, int y, int z)
{
  return z * graph->width * graph->height + y * graph->width + x;
}

/**
 * cudaMallocManaged the graph, and sets every vertex to all 0's
 */
void cudaMallocGraph(Graph **graph, int width, int height, int depth)
{
  // allocate memory for graph and vertices
  unsigned int numVoxels = width * height * depth;
  cudaMallocManaged(graph, sizeof(Graph));
  (**graph).width = width;
  (**graph).height = height;
  (**graph).depth = depth;
  (**graph).numVoxels = numVoxels;
  cudaMallocManaged(&((**graph).voxels), numVoxels * sizeof(Vertex));
  cudaMemset((**graph).voxels, 0, numVoxels * sizeof(Vertex));
}

__device__
int nLinkCapacity(long intensityDiff, unsigned long minIntensity, unsigned long maxIntensity)
{
  return (int) round(25 * exp(-1.0 * intensityDiff * intensityDiff / (2.0 * (maxIntensity - minIntensity) * (maxIntensity - minIntensity))));
}

__device__ __host__
Vertex *getVertex(Graph *graph, int x, int y, int z)
{
  return &(graph->voxels[z * graph->width * graph->height + y * graph->width + x]);
}

__device__ __host__
int getSourceToVoxelFlow(Graph *graph, int x, int y, int z)
{
  Vertex *u = getVertex(graph, x, y, z);
  return u->flows[0];
}

__device__ __host__
int getVoxelToSinkFlow(Graph *graph, int x, int y, int z)
{
  Vertex *u = getVertex(graph, x, y, z);
  return u->flows[1];
}

__device__ __host__
int getVoxelToVoxelFlow(Graph *graph, int x1, int y1, int z1, int x2, int y2, int z2)
{
  if (x1 != x2)
  {
    if (x1 < x2)
    {
      return getVertex(graph, x1, y1, z1)->flows[2];
    } else
    {
      return -getVertex(graph, x2, y2, z2)->flows[2];
    }
  } else if (y1 != y2)
  {
    if (y1 < y2)
    {
      return getVertex(graph, x1, y1, z1)->flows[3];
    } else
    {
      return -getVertex(graph, x2, y2, z2)->flows[3];
    }
  } else if (z1 != z2)
  {
    if (z1 < z2)
    {
      return getVertex(graph, x1, y1, z1)->flows[4];
    } else
    {
      return -getVertex(graph, x2, y2, z2)->flows[4];
    }
  }
  // should never reach this point
  return 0;
}

__device__
void addToSourceToVoxelFlow(Graph *graph, int x, int y, int z, int deltaFlow)
{
  Vertex *u = getVertex(graph, x, y, z);
  u->flows[0] += deltaFlow;
}

__device__
void addToVoxelToSinkFlow(Graph *graph, int x, int y, int z, int deltaFlow)
{
  Vertex *u = getVertex(graph, x, y, z);
  u->flows[1] += deltaFlow;
}

__device__
void addToVoxelToVoxelFlow(Graph *graph, int x1, int y1, int z1, int x2, int y2, int z2, int deltaFlow)
{
  if (x1 != x2)
  {
    if (x1 < x2)
    {
      getVertex(graph, x1, y1, z1)->flows[2] += deltaFlow;
    } else
    {
      getVertex(graph, x2, y2, z2)->flows[2] -= deltaFlow;
    }
  } else if (y1 != y2)
  {
    if (y1 < y2)
    {
      getVertex(graph, x1, y1, z1)->flows[3] += deltaFlow;
    } else
    {
      getVertex(graph, x2, y2, z2)->flows[3] -= deltaFlow;
    }
  } else if (z1 != z2)
  {
    if (z1 < z2)
    {
      getVertex(graph, x1, y1, z1)->flows[4] += deltaFlow;
    } else
    {
      getVertex(graph, x2, y2, z2)->flows[4] -= deltaFlow;
    }
  }
  // should never reach this point
}

__device__ __host__
int getSourceToVoxelCapacity(Graph *graph, int x, int y, int z)
{
  Vertex *u = getVertex(graph, x, y, z);
  return u->capacities[0];
}

__device__ __host__
int getVoxelToSinkCapacity(Graph *graph, int x, int y, int z)
{
  Vertex *u = getVertex(graph, x, y, z);
  return u->capacities[1];
}

__device__ __host__
int getVoxelToVoxelCapacity(Graph *graph, int x1, int y1, int z1, int x2, int y2, int z2)
{
  if (x1 != x2)
  {
    if (x1 > x2)
    {
      int temp = x1;
      x1 = x2;
      x2 = temp;
    }
    Vertex *u = getVertex(graph, x1, y1, z1);
    return u->capacities[2];
  } else if (y1 != y2)
  {
    if (y1 > y2)
    {
      int temp = y1;
      y1 = y2;
      y2 = temp;
    }
    Vertex *u = getVertex(graph, x1, y1, z1);
    return u->capacities[3];
  } else if (z1 != z2)
  {
    if (z1 > z2)
    {
      int temp = z1;
      z1 = z2;
      z2 = temp;
    }
    Vertex *u = getVertex(graph, x1, y1, z1);
    return u->capacities[4];
  }
  // should never reach this point
  return 0;
}

__device__
void push1HandleNeighbor(int heightIndicator, int *pushedOrRelabeled, Graph *graph, int x1, int y1, int z1, int x2, int y2, int z2, int tempExcessIndex)
{
  Vertex *u = getVertex(graph, x1, y1, z1);
  Vertex *v = getVertex(graph, x2, y2, z2);
  if ((u->height[heightIndicator] == v->height[heightIndicator] + 1)
        &&
      (getVoxelToVoxelCapacity(graph, x1, y1, z1, x2, y2, z2) - getVoxelToVoxelFlow(graph, x1, y1, z1, x2, y2, z2) > 0)
      )
  {
    *pushedOrRelabeled = 1;
    int deltaFlow = min(u->excess, getVoxelToVoxelCapacity(graph, x1, y1, z1, x2, y2, z2) - getVoxelToVoxelFlow(graph, x1, y1, z1, x2, y2, z2));
    addToVoxelToVoxelFlow(graph, x1, y1, z1, x2, y2, z2, deltaFlow);
    u->excess -= deltaFlow;
    //printf("in push1HandNeighbor: pushing %d units of flow from (%d,%d,%d) (height %d) to (%d,%d,%d) (height %d), new excess is %u\n", deltaFlow, x1, y1, z1, u->height[heightIndicator], x2, y2, z2, v->height[heightIndicator], u->excess);
    v->tempExcess[tempExcessIndex] = deltaFlow;
  }
}

__global__
void setCapacities(Graph *graph, Volume *volume, unsigned long minIntensity, unsigned long maxIntensity)
{
  int x, y, z;
  for (x = 0; x < volume->width; x++)
  {
    for (y = 0; y < volume->height; y++)
    {
      for (z = 0; z < volume->depth; z++)
      {
        Vertex *u = getVertex(graph, x, y, z);
        unsigned long intensity = getIntensity(volume, x, y, z);
        /** 0 means definitely background, 1 means definitely foreground */
        double score = (double) (intensity - minIntensity + 1) / (maxIntensity - minIntensity + 2);
        u->capacities[0] = (int) (-100.0 * log(1 - score));
        u->capacities[1] = (int) (-100.0 * log(score));
        if (x < volume->width - 1)
        {
          u->capacities[2] = nLinkCapacity(intensity - getIntensity(volume, x + 1, y, z), minIntensity, maxIntensity);
          //(int)(round(50 * exp(-pow(getIntensity(&volume, x, y, z) - getIntensity(&volume, x + 1, y, z), 2) / pow(10, 2))));
        }
        if (y < volume->height - 1)
        {
          u->capacities[3] = nLinkCapacity(intensity - getIntensity(volume, x, y + 1, z), minIntensity, maxIntensity);
          //(int)(round(50 * exp(-pow(getIntensity(&volume, x, y, z) - getIntensity(&volume, x, y + 1, z), 2) / pow(10, 2))));
        }
        if (z < volume->depth - 1)
        {
          u->capacities[4] = nLinkCapacity(intensity - getIntensity(volume, x, y, z + 1), minIntensity, maxIntensity);
          //(int)(round(50 * exp(-pow(getIntensity(&volume, x, y, z) - getIntensity(&volume, x, y, z + 1), 2) / pow(10, 2))));
        }
      }
    }
  }
}

__global__
void preflowInit(Graph *graph)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < graph->numVoxels; i += stride)
  {
    Vertex *u = &(graph->voxels[i]);
    u->flows[0] = u->capacities[0]; // pushes maximum flow from source to u
    u->excess = u->capacities[0];
  }
}

__global__
void push1(int heightIndicator, int *pushedOrRelabeled, Graph *graph)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < graph->numVoxels; i += stride)
  {
    Vertex *u = &(graph->voxels[i]);
    if (u->excess > 0)
    {
      // determine coordinates of voxel that we're pushing from
      int x, y, z;
      x = i % graph->width;
      y = (i / graph->width) % graph->height;
      z = i / (graph->width * graph->height);
      // try on all 8 neighbors of u, it's 6 voxel neighbors and the source and sink neighbors
      // try the source
      if ((u->height[heightIndicator] == SOURCE_HEIGHT(graph) + 1) // u is uphill from the source
            &&
          (getSourceToVoxelCapacity(graph, x, y, z) + getSourceToVoxelFlow(graph, x, y, z) > 0) // residual capacity is positive
                                                  // + sign because we want to subtract flow from voxel to source
        )
      {
        *pushedOrRelabeled = 1;
        int deltaFlow = min(u->excess, getSourceToVoxelCapacity(graph, x, y, z) + getSourceToVoxelFlow(graph, x, y, z));
        addToSourceToVoxelFlow(graph, x, y, z, -deltaFlow); // -deltaFlow because we're pushing flow from here to source
        u->excess -= deltaFlow;
        //printf("in push1: pushing %d units of flow from (%d,%d,%d) to source, new excess is %d\n", deltaFlow, x, y, z, u->excess);
        // don't worry about excess of source
      }
      // try the sink
      if ((u->height[heightIndicator] == SINK_HEIGHT + 1) // u is uphill from the sink
            &&
          (getVoxelToSinkCapacity(graph, x, y, z) - getVoxelToSinkFlow(graph, x, y, z) > 0) // residual capacity is positive
        )
      {
        *pushedOrRelabeled = 1;
        int deltaFlow = min(u->excess, getVoxelToSinkCapacity(graph, x, y, z) - getVoxelToSinkFlow(graph, x, y, z));
        addToVoxelToSinkFlow(graph, x, y, z, deltaFlow);
        u->excess -= deltaFlow;
        //printf("in push1: pushing %d units of flow from (%d,%d,%d) to sink, new excess is %d\n", deltaFlow, x, y, z, u->excess);
        // don't worry about excess of sink;
      }
      // try all the voxel neighbors
      if (x > 0)
      {
        push1HandleNeighbor(heightIndicator, pushedOrRelabeled, graph, x, y, z, x - 1, y, z, 0);
      }
      if (y > 0)
      {
        push1HandleNeighbor(heightIndicator, pushedOrRelabeled, graph, x, y, z, x, y - 1, z, 1);
      }
      if (z > 0)
      {
        push1HandleNeighbor(heightIndicator, pushedOrRelabeled, graph, x, y, z, x, y, z - 1, 2);
      }
      if (x < graph->width - 1)
      {
        push1HandleNeighbor(heightIndicator, pushedOrRelabeled, graph, x, y, z, x + 1, y, z, 3);
      }
      if (y < graph->height - 1)
      {
        push1HandleNeighbor(heightIndicator, pushedOrRelabeled, graph, x, y, z, x, y + 1, z, 4);
      }
      if (z < graph->depth - 1)
      {
        push1HandleNeighbor(heightIndicator, pushedOrRelabeled, graph, x, y, z, x, y, z + 1, 5);
      }
    }
  }
}

__global__
void push2(Graph *graph)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < graph->numVoxels; i += stride)
  {
    Vertex *u = &(graph->voxels[i]);
    for (int i = 0; i < ARRAYLEN(u->tempExcess); i++)
    {
      u->excess += u->tempExcess[i];
      u->tempExcess[i] = 0;
    }
  }
}

__global__
void relabel(int heightIndicator, int *pushedOrRelabeled, Graph *graph)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < graph->numVoxels; i += stride)
  {
    Vertex *u = &(graph->voxels[i]);
    unsigned int uHeight = u->height[heightIndicator];
    u->height[!heightIndicator] = uHeight;
    if (u->excess > 0)
    {
      // test to make sure u is not taller than any of its neighbors
      // simultaneously compute the minimum height of u's neighbors
      unsigned int minHeight = (unsigned int) -1; // max value of unsigned int
      // try source - we assume there's always residual capacity to the source
      if (uHeight > SOURCE_HEIGHT(graph))
      {
        continue;
      } else
      {
        minHeight = min(minHeight, SOURCE_HEIGHT(graph));
      }
      int x, y, z;
      x = i % graph->width;
      y = (i / graph->width) % graph->height;
      z = i / (graph->width * graph->height);
      // try sink
      if (getVoxelToSinkCapacity(graph, x, y, z) - getVoxelToSinkFlow(graph, x, y, z) > 0)
      {
        if (uHeight > SINK_HEIGHT)
        {
          continue;
        } else
        {
          minHeight = min(minHeight, SINK_HEIGHT);
        }
      }
      // try each voxel neighbor
      Vertex *v;
      unsigned int vHeight;
      if (x > 0)
      {
        if (getVoxelToVoxelCapacity(graph, x, y, z, x - 1, y, z) - getVoxelToVoxelFlow(graph, x, y, z, x - 1, y, z) > 0)
        {
          v = getVertex(graph, x - 1, y, z);
          vHeight = v->height[heightIndicator];
          if (uHeight > vHeight)
          {
            continue;
          } else
          {
            minHeight = min(minHeight, vHeight);
          }
        }
      }
      if (y > 0)
      {
        if (getVoxelToVoxelCapacity(graph, x, y, z, x, y - 1, z) - getVoxelToVoxelFlow(graph, x, y, z, x, y - 1, z) > 0)
        {
          v = getVertex(graph, x, y - 1, z);
          vHeight = v->height[heightIndicator];
          if (uHeight > vHeight)
          {
            continue;
          } else
          {
            minHeight = min(minHeight, vHeight);
          }
        }
      }
      if (z > 0)
      {
        if (getVoxelToVoxelCapacity(graph, x, y, z, x, y, z - 1) - getVoxelToVoxelFlow(graph, x, y, z, x, y, z - 1) > 0)
        {
          v = getVertex(graph, x, y, z - 1);
          vHeight = v->height[heightIndicator];
          if (uHeight > vHeight)
          {
            continue;
          } else
          {
            minHeight = min(minHeight, vHeight);
          }
        }
      }
      if (x < graph->width - 1)
      {
        if (getVoxelToVoxelCapacity(graph, x, y, z, x + 1, y, z) - getVoxelToVoxelFlow(graph, x, y, z, x + 1, y, z) > 0)
        {
          v = getVertex(graph, x + 1, y, z);
          vHeight = v->height[heightIndicator];
          if (uHeight > vHeight)
          {
            continue;
          } else
          {
            minHeight = min(minHeight, vHeight);
          }
        }
      }
      if (y < graph->height - 1)
      {
        if (getVoxelToVoxelCapacity(graph, x, y, z, x, y + 1, z) - getVoxelToVoxelFlow(graph, x, y, z, x, y + 1, z) > 0)
        {
          v = getVertex(graph, x, y + 1, z);
          vHeight = v->height[heightIndicator];
          if (uHeight > vHeight)
          {
            continue;
          } else
          {
            minHeight = min(minHeight, vHeight);
          }
        }
      }
      if (z < graph->depth - 1)
      {
        if (getVoxelToVoxelCapacity(graph, x, y, z, x , y, z + 1) - getVoxelToVoxelFlow(graph, x, y, z, x, y, z + 1) > 0)
        {
          v = getVertex(graph, x, y, z + 1);
          vHeight = v->height[heightIndicator];
          if (uHeight > vHeight)
          {
            continue;
          } else
          {
            minHeight = min(minHeight, vHeight);
          }
        }
      }
      u->height[!heightIndicator] = 1 + minHeight;
      *pushedOrRelabeled = 1;
      //printf("in relabel: relabeling vertex (%d,%d,%d) with new height %d\n", x, y, z, 1 + minHeight);
    }
  }
}

/**
 * Given a graph with its max flow computed, store the segmentation result in
 * volume. Foreground = intensity 255, background = intensity 0.
 */
 __global__
void storeSegmentation(Graph *graph, Volume *volume)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < graph->numVoxels; i += stride)
  {
    Vertex *u = &(graph->voxels[i]);
    if (u->capacities[0] - u->flows[0] != 0)
    {
      setIntensity(volume, i, 0, 0, 255);
    } else
    {
      setIntensity(volume, i, 0, 0, 0);
    }
  }
}

void pushRelabel(Graph *graph, Volume *volume)
{
  unsigned long minI = minIntensity(volume), maxI = maxIntensity(volume);
  printf("minI = %lu, maxI = %lu\n", minI, maxI);
  setCapacities<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(graph, volume, minI, maxI);
  printf("set capacities\n");
  int *pushed, *relabeled;
  cudaMallocManaged(&pushed, sizeof(int));
  cudaMallocManaged(&relabeled, sizeof(int));
  preflowInit<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(graph);
  cudaDeviceSynchronize();
  printf("finished preflowInit\n");
  int iteration = 0;
  do
  {
    *pushed = 0;
    *relabeled = 0;
    relabel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>((iteration + 1) % 2, relabeled, graph);
    cudaDeviceSynchronize();
    //printf("  finished relabel iteration %d: %s relabel\n", iteration, *relabeled ? "did" : "did not");
    push1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(iteration % 2, pushed, graph);
    cudaDeviceSynchronize();
    //printf("  finished push1 iteration %d: %s push\n", iteration, *pushed ? "did" : "did not");
    push2<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(graph);
    cudaDeviceSynchronize();
    //printf("  finished push2 iteration %d\n", iteration);
    //printf("  finished iteration %d\n", iteration);
    //printf("GRAPH:\n");
    //printGraph(graph, iteration % 2);
    //printf("  after iteration %d, *pushed = %d and *relabeled = %d\n", iteration, *pushed, *relabeled);
    iteration++;
  } while (*pushed || *relabeled);
  printf("finished pushRelabel\n");
  cudaFree(pushed);
  cudaFree(relabeled);
}

void printGraph(Graph *graph)
{
  int x, y, z;
  for (z = 0; z < graph->depth; z++)
  {
    for (y = 0; y < graph->height; y++)
    {
      for (x = 0; x < graph->width; x++)
      {
        printf("for x=%d, y=%d, z=%d:\n", x, y, z);
        Vertex *u = getVertex(graph, x, y, z);
        //printf("  heights:                  %u, %u\n", u->height[0], u->height[1]);
        //printf("  excess:                   %u\n", u->excess);
        printf("  source to voxel capacity: %d\n", getSourceToVoxelCapacity(graph, x, y, z));
        printf("  source to voxel flow:     %d\n", getSourceToVoxelFlow(graph, x, y, z));
        printf("  voxel to sink capacity:   %d\n", getVoxelToSinkCapacity(graph, x, y, z));
        printf("  voxel to sink flow:       %d\n", getVoxelToSinkFlow(graph, x, y, z));
        if (x > 0)
        {
          printf("  capacity to (%d, %d, %d):    %d\n", x - 1, y, z, getVoxelToVoxelCapacity(graph, x, y, z, x - 1, y, z));
          printf("  flow to (%d, %d, %d):        %d\n", x - 1, y, z, getVoxelToVoxelFlow(graph, x, y, z, x - 1, y, z));
        }
        if (y > 0)
        {
          printf("  capacity to (%d, %d, %d):    %d\n", x, y - 1, z, getVoxelToVoxelCapacity(graph, x, y, z, x, y - 1, z));
          printf("  flow to (%d, %d, %d):        %d\n", x, y - 1, z, getVoxelToVoxelFlow(graph, x, y, z, x, y - 1, z));
        }
        if (z > 0)
        {
          printf("  capacity to (%d, %d, %d):    %d\n", x, y, z - 1, getVoxelToVoxelCapacity(graph, x, y, z, x, y, z - 1));
          printf("  flow to (%d, %d, %d):        %d\n", x, y, z - 1, getVoxelToVoxelFlow(graph, x, y, z, x, y, z - 1));
        }
        if (x < graph->width - 1)
        {
          printf("  capacity to (%d, %d, %d):    %d\n", x + 1, y, z, getVoxelToVoxelCapacity(graph, x, y, z, x + 1, y, z));
          printf("  flow to (%d, %d, %d):        %d\n", x + 1, y, z, getVoxelToVoxelFlow(graph, x, y, z, x + 1, y, z));
        }
        if (y < graph->height - 1)
        {
          printf("  capacity to (%d, %d, %d):    %d\n", x, y + 1, z, getVoxelToVoxelCapacity(graph, x, y, z, x, y + 1, z));
          printf("  flow to (%d, %d, %d):        %d\n", x, y + 1, z, getVoxelToVoxelFlow(graph, x, y, z, x, y + 1, z));
        }
        if (z < graph->depth - 1)
        {
          printf("  capacity to (%d, %d, %d):    %d\n", x, y, z + 1, getVoxelToVoxelCapacity(graph, x, y, z, x, y, z + 1));
          printf("  flow to (%d, %d, %d):        %d\n", x, y, z + 1, getVoxelToVoxelFlow(graph, x, y, z, x, y, z + 1));
        }
      }
    }
  }
}

/**
 * Returns 1 if the flow in the given graph is indeed the maximum flow
 */
int isMaxFlow(Graph *graph)
{
  /**
   * For eaxh voxel, first verify that if there is residual capacity from the
   * source then there is none to the sink. Then verify that if there is
   * residual capacity from a voxel to its neighbor then it is not the case that
   * one has residual capacity from the source and one has residual capacity to
   * the sink.
   *
   * Also verify that the excess is zero for each voxel.
   */
  int x, y, z;
  for (z = 0; z < graph->depth; z++)
  {
    for (y = 0; y < graph->height; y++)
    {
      for (x = 0; x < graph->width; x++)
      {
        Vertex *u = getVertex(graph, x, y, z);
        if (u->excess > 0)
        {
          printf("VERIFICATION FAILED at (%d,%d,%d): excess = %u\n", x, y, z, u->excess);
          return 0;
        }
        char visited[graph->numVoxels];
        bzero(visited, graph->numVoxels * sizeof(char));
        if (u->capacities[0] - u->flows[0] > 0)
        {
          if (pathfinderToSink(graph, x, y, z, visited))
          {
            printf("VERIFICATION FAILED: path from source to sink starting at voxel (%d,%d,%d)\n", x, y, z);
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

int pathfinderToSink(Graph *graph, int x, int y, int z, char visited[])
{
  if (getVoxelToSinkCapacity(graph, x, y, z) - getVoxelToSinkFlow(graph, x, y, z) > 0)
  {
    printf("reverse path:\n");
    printf("sink\n");
    printf("(%d, %d, %d)\n", x, y, z);
    return 1;
  }
  visited[getIndex(graph, x, y, z)] = 1;
  if (x > 0)
  {
    if (!visited[getIndex(graph, x - 1, y, z)] && (getVoxelToVoxelCapacity(graph, x, y, z, x - 1, y, z) - getVoxelToVoxelFlow(graph, x, y, z, x - 1, y, z) > 0))
    {
      if (pathfinderToSink(graph, x - 1, y, z, visited))
      {
        printf("(%d, %d, %d)", x, y, z);
        return 1;
      }
    }
  }
  if (y > 0)
  {
    if (!visited[getIndex(graph, x, y - 1, z)] && (getVoxelToVoxelCapacity(graph, x, y, z, x, y - 1, z) - getVoxelToVoxelFlow(graph, x, y, z, x, y - 1, z) > 0))
    {
      if (pathfinderToSink(graph, x, y - 1, z, visited))
      {
        printf("(%d, %d, %d)\n", x, y, z);
        return 1;
      }
    }
  }
  if (z > 0)
  {
    if (!visited[getIndex(graph, x, y, z - 1)] && (getVoxelToVoxelCapacity(graph, x, y, z, x, y, z - 1) - getVoxelToVoxelFlow(graph, x, y, z, x, y, z - 1) > 0))
    {
      if (pathfinderToSink(graph, x, y, z - 1, visited))
      {
        printf("(%d, %d, %d)\n", x, y, z);
        return 1;
      }
    }
  }
  if (x < graph->width - 1)
  {
    if (!visited[getIndex(graph, x + 1, y, z)] && (getVoxelToVoxelCapacity(graph, x, y, z, x + 1, y, z) - getVoxelToVoxelFlow(graph, x, y, z, x + 1, y, z) > 0))
    {
      if (pathfinderToSink(graph, x + 1, y, z, visited))
      {
        printf("(%d, %d, %d)\n", x, y, z);
        return 1;
      }
    }
  }
  if (y < graph->height - 1)
  {
    if (!visited[getIndex(graph, x, y + 1, z)] && (getVoxelToVoxelCapacity(graph, x, y, z, x, y + 1, z) - getVoxelToVoxelFlow(graph, x, y, z, x, y + 1, z) > 0))
    {
      if (pathfinderToSink(graph, x, y + 1, z, visited))
      {
        printf("(%d, %d, %d)\n", x, y, z);
        return 1;
      }
    }
  }
  if (z < graph->depth - 1)
  {
    if (!visited[getIndex(graph, x, y, z + 1)] && (getVoxelToVoxelCapacity(graph, x, y, z, x, y, z + 1) - getVoxelToVoxelFlow(graph, x, y, z, x, y, z + 1) > 0))
    {
      if (pathfinderToSink(graph, x, y, z + 1, visited))
      {
        printf("(%d, %d, %d)\n", x, y, z);
        return 1;
      }
    }
  }
  return 0;
}
