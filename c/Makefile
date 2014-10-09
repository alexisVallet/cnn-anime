CC=gcc

all: gdt.so

gdt.so: gdt.o
	$(CC) -shared -Wl,-soname,gdt.so -o gdt.so gdt.o

gdt.o: gdt.c
	$(CC) -fPIC -O3 -c gdt.c -o gdt.o
