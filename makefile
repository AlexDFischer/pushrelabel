CC = nvcc
TARGET = pushrelabel

#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR = src
INCDIR = include
BUILDDIR = bin

#Flags, Libraries and Includes
LIBTIFFFLAGS = -I/home/afis/lib/libtiff/include -L/home/afis/lib/libtiff/lib -ltiff
LIB = -lm $(LIBTIFFFLAGS)
CFLAGS = --compiler-options -Wall -I$(INCDIR) $(LIB)

OBJECTS = volume.o graph.o
HEADERS = project.h volume.h graph.h

$(TARGET): $(SRCDIR)/$(TARGET).cu $(addprefix $(BUILDDIR)/, $(OBJECTS))
	$(CC) -o bin/$@ $^ $(CFLAGS)

writetiff: $(SRCDIR)/writetiff.cu $(addprefix $(BUILDDIR)/, $(OBJECTS))
	$(CC) -o bin/$@ $^ $(CFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(addprefix $(INCDIR)/, $(HEADERS))
	$(CC) -c --device-c -o $@ $< $(CFLAGS)

clean:
	rm $(addprefix $(BUILDDIR)/, $(OBJECTS) $(TARGET) writetiff)
