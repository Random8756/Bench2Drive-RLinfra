#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// gcc -Wall -Wextra -O2 -shared -fPIC -o tools/fake_bind.so tools/fake_bind.c -ldl

static char override_ip[INET_ADDRSTRLEN] = "127.0.0.2";

__attribute__((constructor))
void load_override_ip() {
    const char *env_ip = getenv("FAKE_BIND_IP");
    if (env_ip && strlen(env_ip) < sizeof(override_ip)) {
        strncpy(override_ip, env_ip, sizeof(override_ip));
    }
}

int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    static int (*real_bind)(int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_bind) {
        real_bind = dlsym(RTLD_NEXT, "bind");
    }

    if (addr->sa_family == AF_INET) {
        struct sockaddr_in *in = (struct sockaddr_in *)addr;
        inet_pton(AF_INET, override_ip, &in->sin_addr);
    }

    return real_bind(sockfd, addr, addrlen);
}
