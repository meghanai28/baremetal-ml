CC = gcc
CFLAGS = -Wall -Wextra -O0

linreg: linreg.c
	$(CC) $(CFLAGS) linreg.c -o linreg

clean:
	rm -f linreg