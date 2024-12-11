CFLAGS=-L /opt/homebrew/lib -I /opt/homebrew/include

all : inference

inference : inference.c
	gcc $(CFLAGS) inference.c -o inference -lonnxruntime
