CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L
DEBUG_FLAGS = -O0
RELEASE_FLAGS = -O3 -march=native

TARGET = linreg_naive
SRC = linreg_naive.c

all: release

debug:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(SRC) -o $(TARGET)

release:
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)