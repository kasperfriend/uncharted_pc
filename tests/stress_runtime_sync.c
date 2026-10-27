/* Guest semaphores under concurrent use: waiters that time out, signals that arrive while a
 * waiter is being woken, a delete with waiters present.
 *
 * tests/test_runtime_sync.c checks the contract of one call at a time. This one drives the same
 * code from 8 waiting threads and 2 signalling threads over 4 semaphores for a second with mixed
 * timeouts, which is where the bookkeeping of the per-thread condition variable and of the FIFO
 * list shows: a waiter that is woken by a signal instead of its deadline, a deadline that fires
 * for a waiter another thread just signalled, and list surgery on both ends at once. It is also
 * the target of the ThreadSanitizer build in tools/ci/linux-tests.sh.
 *
 * Self-contained apart from the two runtime files it links (src/runtime_sema.c).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "runtime.h"

unsigned runtime_sema_waiters(uint32_t id);
uintptr_t runtime_sema_resolve(const char *name);

static int32_t (*sem_wait_fn)(uint32_t, int32_t, uint32_t *);
static int32_t (*sem_signal_fn)(uint32_t, int32_t);
static int32_t (*sem_delete_fn)(uint32_t);
static uint32_t semaphores[4];
static int stop;
static long waits, timeouts, acquired;

static void *waiting_thread(void *arg) {
    (void)arg;
    unsigned long n = 0;
    while (!__atomic_load_n(&stop, __ATOMIC_ACQUIRE)) {
        uint32_t timeout_us = 200;
        const uint32_t id = semaphores[(n++) & 3];
        const int32_t result = sem_wait_fn(id, 1, &timeout_us);
        __atomic_add_fetch(&waits, 1, __ATOMIC_RELAXED);
        if (result == 0) {
            __atomic_add_fetch(&acquired, 1, __ATOMIC_RELAXED);
        } else if (result == (int32_t)0x8002003c) {
            __atomic_add_fetch(&timeouts, 1, __ATOMIC_RELAXED);
        } else if (result != (int32_t)0x8002000d) { /* deleted while waiting: expected at the end */
            fprintf(stderr, "FAIL: unexpected wait result %#x\n", (unsigned)result);
            abort();
        }
    }
    return NULL;
}

static void *signalling_thread(void *arg) {
    (void)arg;
    unsigned long n = 0;
    while (!__atomic_load_n(&stop, __ATOMIC_ACQUIRE)) {
        sem_signal_fn(semaphores[(n++) & 3], 1);
        const struct timespec pause = {0, 100000}; /* 100 us: the waiters' deadline is 200 us */
        nanosleep(&pause, NULL);
    }
    return NULL;
}

int main(void) {
    int32_t (*create)(uint32_t *, const char *, uint32_t, int32_t, int32_t, const void *) =
        (void *)runtime_sema_resolve("188x57JYp0g#p#J");
    sem_wait_fn = (void *)runtime_sema_resolve("Zxa0VhQVTsk#p#J");
    sem_signal_fn = (void *)runtime_sema_resolve("4czppHBiriw#p#J");
    sem_delete_fn = (void *)runtime_sema_resolve("R1Jvn8bSCW8#p#J");
    if (!create || !sem_wait_fn || !sem_signal_fn || !sem_delete_fn) {
        fprintf(stderr, "FAIL: semaphore exports not found\n");
        return 1;
    }
    for (unsigned i = 0; i < 4; ++i) {
        semaphores[i] = 0;
        if (create(&semaphores[i], "stress", 1, 0, 4096, NULL) != 0) {
            fprintf(stderr, "FAIL: semaphore %u not created\n", i);
            return 1;
        }
    }
    pthread_t threads[10];
    for (unsigned i = 0; i < 8; ++i) {
        pthread_create(&threads[i], NULL, waiting_thread, NULL);
    }
    for (unsigned i = 8; i < 10; ++i) {
        pthread_create(&threads[i], NULL, signalling_thread, NULL);
    }
    const struct timespec run = {1, 0};
    nanosleep(&run, NULL);
    __atomic_store_n(&stop, 1, __ATOMIC_RELEASE);
    for (unsigned i = 8; i < 10; ++i) {
        pthread_join(threads[i], NULL);
    }
    /* Release whoever is still blocked in a wait. */
    for (unsigned i = 0; i < 4; ++i) {
        sem_signal_fn(semaphores[i], 4096);
    }
    for (unsigned i = 0; i < 8; ++i) {
        pthread_join(threads[i], NULL);
    }
    unsigned left = 0;
    for (unsigned i = 0; i < 4; ++i) {
        left += runtime_sema_waiters(semaphores[i]);
    }
    printf("sema stress: %ld waits (%ld acquired, %ld timed out), waiters left %u\n",
           __atomic_load_n(&waits, __ATOMIC_RELAXED), __atomic_load_n(&acquired, __ATOMIC_RELAXED),
           __atomic_load_n(&timeouts, __ATOMIC_RELAXED), left);
    if (left != 0) {
        fprintf(stderr, "FAIL: %u waiters left behind\n", left);
    }
    for (unsigned i = 0; i < 4; ++i) {
        sem_delete_fn(semaphores[i]);
    }
    return left != 0;
}
