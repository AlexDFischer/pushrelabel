#include "project.h"
#include "volume.h"

char *programName;

int main(int argc, char *argv[])
{
  programName = argv[0];
  if (argc < 2)
  {
    printf("Usage:\n");
    printf("  %s inputRawVolume outputTiffDir\n", programName);
  }
  // read our raw file into volume
  Volume *volume;
  readRaw(&volume, argv[1]);
  printf("read volume from raw file. dimensions = (%d,%d,%d)\n", volume->width, volume->height, volume->depth);
  writeTiff(volume, argv[2]);
  printf ("wrote volume to tiff directory %s\n", argv[2]);
}
