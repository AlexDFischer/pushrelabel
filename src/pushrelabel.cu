#include "project.h"
#include "volume.h"
#include "graph.h"

char *programName;

int main(int argc, char *argv[])
{
  programName = argv[0];
  if (argc < 2)
  {
    printf("Usage:\n");
    printf("  %s inputRawVolume outputRawVolume\n", programName);
    exit(0);
  }
  // read our raw file into volume
  Volume *volume;
  readRaw(&volume, argv[1]);
  printf("read volume from raw file. dimensions = (%d,%d,%d)\n", volume->width, volume->height, volume->depth);
  Graph *graph;
  cudaMallocGraph(&graph, volume->width, volume->height, volume->depth);
  //printGraph(graph);
  pushRelabel(graph, volume);
  //printGraph(graph);
  if (isMaxFlow(graph))
  {
    printf("flow verification succeeded: we found the max flow\n");
  } else
  {
    printf("flow verification failed\n");
  }
  cudaFree(volume->data);
  // resize volume data so we only have to store 1 byte per voxel
  volume->bytesPerPixel = 1;
  cudaMallocManagedVolume(volume);
  storeSegmentation<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(graph, volume);
  cudaDeviceSynchronize();
  writeRaw(volume, argv[2]);
  printf("wrote segmentation to %s\n", argv[2]);
  cudaFree(volume->data);
  cudaFree(volume);
  cudaFree(&(graph->voxels));
  cudaFree(graph);
}
