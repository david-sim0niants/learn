#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/stat.h>
#include <signal.h>

#include <systemd/sd-journal.h>

void daemonize();
void wait_for_exit_signals();

int main()
{
    // daemonize();

    const char *daemon_name = "Primary";
#if DAEMON_SECONDARY
    daemon_name = "Secondary";
#endif

    sd_journal_print(LOG_NOTICE, "DAEMON START: %s", daemon_name);
    wait_for_exit_signals();
    sd_journal_print(LOG_NOTICE, "DAEMON END: %s", daemon_name);
    return 0;
}

void daemonize()
{
    pid_t pid;

    pid = fork();
    if (pid < 0) {
        perror("fork() failed");
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    if (setsid() < 0) {
        perror("setsid() failed");
        exit(EXIT_FAILURE);
    }

    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    pid = fork();
    if (pid < 0) {
        perror("fork() failed");
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    umask(0);
    chdir("/");

    for (int x = sysconf(_SC_OPEN_MAX); x >= 0; --x)
        close(x);
}

void wait_for_exit_signals()
{
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGTERM);
    sigaddset(&sigset, SIGINT);
    sigprocmask(SIG_BLOCK, &sigset, NULL);

    int signal;
    sigwait(&sigset, &signal);

    if (signal == SIGTERM)
        sd_journal_print(LOG_NOTICE, "Got SIGTERM");
    else if (signal == SIGINT)
        sd_journal_print(LOG_NOTICE, "Got SIGINT");
    else
        sd_journal_print(LOG_WARNING, "Got unexpected signal");
}
