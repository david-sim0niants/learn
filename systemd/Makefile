DAEMON_SRC = daemon.c
DEPS = -lsystemd

PREFIX = /usr/local
BINDIR = $(PREFIX)/bin

.PHONY: all clean install

all: daemon-primary daemon-secondary

clean:
	rm daemon-primary daemon-secondary

install: daemon-primary daemon-secondary
	install $? $(BINDIR)

daemon-primary: $(DAEMON_SRC)
	$(CC) $(DAEMON_SRC) -o $@ $(DEPS)

daemon-secondary: $(DAEMON_SRC)
	$(CC) $(DAEMON_SRC) -o $@ $(DEPS) -DDAEMON_SECONDARY=1
