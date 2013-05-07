CC = g++
GCC = gcc
CFLAGS = -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: SENNA_utils.o SENNA_Hash.o SENNA_Tokenizer.o rnntag

SENNA_utils.o : SENNA_utils.c
	$(GCC) $(CFLAGS) $(OPT_DEF) -c SENNA_utils.c 

SENNA_Hash.o : SENNA_Hash.c
	$(GCC) $(CFLAGS) $(OPT_DEF) -c SENNA_Hash.c

SENNA_Tokenizer.o : SENNA_Tokenizer.c
	$(GCC) $(CFLAGS) $(OPT_DEF) -c SENNA_Tokenizer.c

rnntag : myrnn.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) myrnn.cpp SENNA_Hash.o SENNA_Tokenizer.o SENNA_utils.o -fopenmp -DLINUX -o rnntag 

clean:
	rm -rf *.o rnntag
