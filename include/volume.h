#include "project.h"
#include "tiffio.h"

struct Volume
{
  int width, height, depth, pixelFormat, bytesPerPixel;
  char *data;
};

typedef struct Volume Volume;

__host__ __device__
unsigned long getIntensity(Volume *volume, int x, int y, int z);


__host__ __device__
void setIntensity(Volume *volume, int x, int y, int z, unsigned long intensity);

void cudaMallocManagedVolume(Volume *volume);

void mallocVolume(Volume *volume);

unsigned long maxIntensity(Volume *volume);

unsigned long minIntensity(Volume *volume);

void printVolume(Volume *volume);

int readRaw(Volume **volume, char *fileName);

int writeRaw(Volume *volume, char *fileName);

int writeTiff(Volume *volume, char *dirName);
