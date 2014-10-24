CC=gcc

all: gdt.so jpeg_decode.so

clean:
	rm *.o *.so

gdt.so: gdt.o
	$(CC) -shared -Wl,-soname,gdt.so -o gdt.so gdt.o

gdt.o: gdt.c
	$(CC) -fPIC -O3 -c gdt.c -o gdt.o

jpeg_decode.so: jpeg_decode.o
	$(CC) -shared -Wl,--no-as-needed -lturbojpeg -Wl,-soname,jpeg_decode.so -o jpeg_decode.so jpeg_decode.o

jpeg_decode.o: jpeg_decode.c
	$(CC) -fPIC -O3 -c jpeg_decode.c -o jpeg_decode.o
