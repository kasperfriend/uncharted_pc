/* Guest synchronization primitives (src/runtime_sema.c, src/runtime_rwlock.c).
 *
 * Both files are self-contained C (libc and pthreads only), so this test links them on their own
 * and drives them through the same resolver the guest import table uses. It covers what the
 * port depends on beyond the happy path: many live objects (the registries are hashed, so a
 * lookup must not depend on the list it replaced), FIFO waiter order, timeouts, cancel, delete
 * with a blocked waiter, and the rwlock owner/EBUSY/EPERM rules.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Same encoding as the runtime: 0x80020000 | errno. */
#define ERR(n) ((int32_t)(UINT32_C(0x80020000) | (n)))
#define OK(expr)                                                                                   \
    do {                                                                                           \
        if ((expr) != 0) {                                                                         \
            fprintf(stderr, "FAIL %s:%d: %s returned %#x\n", __FILE__, __LINE__, #expr,            \
                    (unsigned)(expr));                                                             \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

extern uintptr_t runtime_sema_resolve(const char *name);
extern uintptr_t runtime_rwlock_resolve(const char *name);

typedef int32_t (*SemCreate)(uint32_t *, const char *, uint32_t, int32_t, int32_t, const void *);
typedef int32_t (*SemWait)(uint32_t, int32_t, uint32_t *);
typedef int32_t (*SemPoll)(uint32_t, int32_t);
typedef int32_t (*SemSignal)(uint32_t, int32_t);
typedef int32_t (*SemCancel)(uint32_t, int32_t, int32_t *);
typedef int32_t (*SemDelete)(uint32_t);

typedef struct {
    int64_t seconds, nanoseconds;
} GuestTime;

typedef int32_t (*RwInit)(void **, void *const *, const char *);
typedef int32_t (*RwHandleOp)(void **);
typedef int32_t (*RwTimedOp)(void **, const GuestTime *);

static int failures;

static void *resolve(uintptr_t (*resolver)(const char *), const char *nid) {
    const uintptr_t address = resolver(nid);
    if (!address) {
        fprintf(stderr, "FAIL: %s not resolved\n", nid);
        ++failures;
    }
    return (void *)address;
}

/* --- semaphores ---------------------------------------------------------------------------- */

static SemWait sem_wait_fn;
static SemPoll sem_poll_fn;
static SemSignal sem_signal_fn;
static SemDelete sem_delete_fn;

struct WaiterArgs {
    uint32_t id;
    int32_t need;
    uint32_t timeout_us;
    int timed;
    int32_t result;
};

static void *wait_thread(void *arg) {
    struct WaiterArgs *waiter = arg;
    const int32_t result = sem_wait_fn(waiter->id, waiter->need,
                                       waiter->timed ? &waiter->timeout_us : NULL);
    __atomic_store_n(&waiter->result, result, __ATOMIC_RELEASE);
    return NULL;
}

static void test_semaphores(SemCreate sem_create) {
    /* Many live objects: the registry has to find each of them by id, not by position. */
    enum { COUNT = 64 };
    uint32_t ids[COUNT];
    for (int i = 0; i < COUNT; ++i) {
        ids[i] = 0;
        if (sem_create(&ids[i], "test", 1, i + 1, 4096, NULL) != 0 || ids[i] == 0) {
            fprintf(stderr, "FAIL: semaphore %d not created\n", i);
            ++failures;
            return;
        }
    }
    for (int i = 0; i < COUNT; ++i) {
        if (sem_poll_fn(ids[i], i + 1) != 0) {
            fprintf(stderr, "FAIL: poll of semaphore %d (count %d)\n", i, i + 1);
            ++failures;
        }
        /* The token was taken: it is gone now. */
        if (sem_poll_fn(ids[i], 1) != ERR(16)) {
            fprintf(stderr, "FAIL: second poll of semaphore %d did not report EBUSY\n", i);
            ++failures;
        }
        OK(sem_signal_fn(ids[i], 2));
        if (sem_poll_fn(ids[i], 2) != 0) {
            fprintf(stderr, "FAIL: signaled semaphore %d did not hand out 2 tokens\n", i);
            ++failures;
        }
    }
    for (int i = 0; i < COUNT; i += 2) {
        OK(sem_delete_fn(ids[i]));
    }
    /* Deleting twice reports "no such semaphore" for the deleted ones, and the rest still work. */
    if (sem_delete_fn(ids[0]) != ERR(3)) {
        fprintf(stderr, "FAIL: deleting a deleted semaphore did not report ERR(3)\n");
        ++failures;
    }
    for (int i = 1; i < COUNT; i += 2) {
        OK(sem_signal_fn(ids[i], 1));
        OK(sem_poll_fn(ids[i], 1));
        OK(sem_delete_fn(ids[i]));
    }

    /* A blocked waiter is woken by a signal, in FIFO order, one token each. */
    uint32_t id = 0;
    if (sem_create(&id, "fifo", 1, 0, 16, NULL) != 0) {
        fprintf(stderr, "FAIL: fifo semaphore not created\n");
        ++failures;
        return;
    }
    enum { WAITERS = 4 };
    struct WaiterArgs waiters[WAITERS];
    pthread_t threads[WAITERS];
    for (int i = 0; i < WAITERS; ++i) {
        waiters[i] = (struct WaiterArgs){.id = id, .need = 1};
        pthread_create(&threads[i], NULL, wait_thread, &waiters[i]);
        usleep(2000); /* in creation order: the FIFO list is in that order too */
    }
    for (int i = 0; i < WAITERS; ++i) {
        OK(sem_signal_fn(id, 1));
        usleep(2000);
    }
    for (int i = 0; i < WAITERS; ++i) {
        pthread_join(threads[i], NULL);
        if (__atomic_load_n(&waiters[i].result, __ATOMIC_ACQUIRE) != 0) {
            fprintf(stderr, "FAIL: waiter %d returned %#x\n", i, (unsigned)waiters[i].result);
            ++failures;
        }
    }
    OK(sem_delete_fn(id));

    /* Timeout: nobody signals, the wait reports ETIMEDOUT after the given microseconds. */
    id = 0;
    if (sem_create(&id, "timeout", 1, 0, 4, NULL) != 0) {
        fprintf(stderr, "FAIL: timeout semaphore not created\n");
        ++failures;
        return;
    }
    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);
    struct WaiterArgs timed = {.id = id, .need = 1, .timed = 1, .timeout_us = 20000};
    pthread_t thread;
    pthread_create(&thread, NULL, wait_thread, &timed);
    pthread_join(thread, NULL);
    clock_gettime(CLOCK_MONOTONIC, &after);
    const long elapsed_us =
        (after.tv_sec - before.tv_sec) * 1000000L + (after.tv_nsec - before.tv_nsec) / 1000L;
    if (__atomic_load_n(&timed.result, __ATOMIC_ACQUIRE) != ERR(60)) {
        fprintf(stderr, "FAIL: timed wait returned %#x, expected ETIMEDOUT\n",
                (unsigned)timed.result);
        ++failures;
    }
    if (elapsed_us < 15000 || elapsed_us > 400000) {
        fprintf(stderr, "FAIL: 20 ms wait took %ld us\n", elapsed_us);
        ++failures;
    }
    OK(sem_delete_fn(id));

    /* Delete with a blocked waiter: the waiter gets EKILL(13) and the semaphore is freed. */
    id = 0;
    if (sem_create(&id, "delete", 1, 0, 4, NULL) != 0) {
        fprintf(stderr, "FAIL: delete semaphore not created\n");
        ++failures;
        return;
    }
    struct WaiterArgs blocked = {.id = id, .need = 1};
    pthread_create(&thread, NULL, wait_thread, &blocked);
    usleep(5000);
    OK(sem_delete_fn(id));
    pthread_join(thread, NULL);
    if (__atomic_load_n(&blocked.result, __ATOMIC_ACQUIRE) != ERR(13)) {
        fprintf(stderr, "FAIL: waiter of a deleted semaphore returned %#x\n",
                (unsigned)blocked.result);
        ++failures;
    }

    /* Cancel: wakes every waiter with EABRT(85) and resets the count. */
    id = 0;
    if (sem_create(&id, "cancel", 1, 0, 8, NULL) != 0) {
        fprintf(stderr, "FAIL: cancel semaphore not created\n");
        ++failures;
        return;
    }
    SemCancel sem_cancel_fn = (SemCancel)resolve(runtime_sema_resolve, "4DM06U2BNEY#p#J");
    struct WaiterArgs cancelled = {.id = id, .need = 1};
    pthread_create(&thread, NULL, wait_thread, &cancelled);
    usleep(5000);
    int32_t woken = 0;
    OK(sem_cancel_fn(id, 3, &woken));
    pthread_join(thread, NULL);
    if (__atomic_load_n(&cancelled.result, __ATOMIC_ACQUIRE) != ERR(85) || woken != 1) {
        fprintf(stderr, "FAIL: cancel result %#x, woken %d\n", (unsigned)cancelled.result, woken);
        ++failures;
    }
    OK(sem_poll_fn(id, 3));
    OK(sem_delete_fn(id));
}

/* --- rwlocks ------------------------------------------------------------------------------- */

static RwInit rw_init_fn;
static RwHandleOp rw_read_fn, rw_write_fn, rw_tryread_fn, rw_trywrite_fn, rw_unlock_fn,
    rw_destroy_fn;
static RwTimedOp rw_timedread_fn;

/* `result` and `release` are handed between the test's threads, which is synchronized by the
 * guest lock under test and by the usual "give the thread time" sleeps -- so they are accessed
 * atomically: otherwise ThreadSanitizer reports the plain reads below as races. */
struct RwArgs {
    void *handle;
    int32_t result;
};

static void *read_thread(void *arg) {
    struct RwArgs *args = arg;
    const int32_t result = rw_read_fn(&args->handle);
    __atomic_store_n(&args->result, result, __ATOMIC_RELEASE);
    if (result == 0) {
        rw_unlock_fn(&args->handle);
    }
    return NULL;
}

struct WriterArgs {
    void *handle;
    int release;
    int32_t result;
};

static void *writer_thread(void *arg) {
    struct WriterArgs *args = arg;
    const int32_t result = rw_write_fn(&args->handle);
    __atomic_store_n(&args->result, result, __ATOMIC_RELEASE);
    if (result == 0) {
        while (!__atomic_load_n(&args->release, __ATOMIC_ACQUIRE)) {
            usleep(1000);
        }
        rw_unlock_fn(&args->handle);
    }
    return NULL;
}

struct Shared {
    void *handle;
    pthread_mutex_t lock;
    int readers;
};

static void *shared_read_thread(void *arg) {
    struct Shared *shared = arg;
    if (rw_read_fn(&shared->handle) == 0) {
        pthread_mutex_lock(&shared->lock);
        const int now = ++shared->readers;
        pthread_mutex_unlock(&shared->lock);
        usleep(2000);
        pthread_mutex_lock(&shared->lock);
        --shared->readers;
        pthread_mutex_unlock(&shared->lock);
        (void)now;
        rw_unlock_fn(&shared->handle);
    }
    return NULL;
}

static void test_rwlocks(void) {
    /* Many live locks, all initialized and used in turn. */
    enum { COUNT = 32 };
    void *handles[COUNT];
    memset(handles, 0, sizeof(handles));
    for (int i = 0; i < COUNT; ++i) {
        if (rw_init_fn(&handles[i], NULL, "rw") != 0) {
            fprintf(stderr, "FAIL: rwlock %d not initialized\n", i);
            ++failures;
            return;
        }
    }
    for (int i = 0; i < COUNT; ++i) {
        OK(rw_tryread_fn(&handles[i]));
        OK(rw_unlock_fn(&handles[i]));
        OK(rw_trywrite_fn(&handles[i]));
        OK(rw_unlock_fn(&handles[i]));
    }
    for (int i = 0; i < COUNT; ++i) {
        OK(rw_destroy_fn(&handles[i]));
        if (handles[i] != (void *)(uintptr_t)1) {
            fprintf(stderr, "FAIL: destroyed rwlock %d kept its handle\n", i);
            ++failures;
        }
        /* A destroyed handle is no longer in the registry. */
        if (rw_read_fn(&handles[i]) != ERR(22)) {
            fprintf(stderr, "FAIL: read of destroyed rwlock %d did not report EINVAL\n", i);
            ++failures;
        }
    }

    /* Owner rules: a read lock this thread holds blocks a write lock (EDEADLK, EBUSY when trying),
     * an unlock without the lock is EPERM, and a held lock cannot be destroyed. */
    void *handle = NULL;
    if (rw_init_fn(&handle, NULL, "owner") != 0) {
        fprintf(stderr, "FAIL: owner rwlock not initialized\n");
        ++failures;
        return;
    }
    OK(rw_read_fn(&handle));
    if (rw_write_fn(&handle) != ERR(11)) {
        fprintf(stderr, "FAIL: write lock while holding a read lock did not report EDEADLK\n");
        ++failures;
    }
    if (rw_trywrite_fn(&handle) != ERR(16)) {
        fprintf(stderr, "FAIL: try-write while holding a read lock did not report EBUSY\n");
        ++failures;
    }
    if (rw_destroy_fn(&handle) != ERR(16)) {
        fprintf(stderr, "FAIL: destroying a held rwlock did not report EBUSY\n");
        ++failures;
    }
    OK(rw_unlock_fn(&handle));
    if (rw_unlock_fn(&handle) != ERR(1)) {
        fprintf(stderr, "FAIL: unlock without a lock did not report EPERM\n");
        ++failures;
    }
    /* A writer excludes another thread's read lock until it unlocks. */
    struct RwArgs reader = {.handle = handle};
    pthread_t thread;
    OK(rw_write_fn(&handle));
    pthread_create(&thread, NULL, read_thread, &reader);
    usleep(5000);
    OK(rw_unlock_fn(&handle));
    pthread_join(thread, NULL);
    if (__atomic_load_n(&reader.result, __ATOMIC_ACQUIRE) != 0) {
        fprintf(stderr, "FAIL: reader behind a writer returned %#x\n", (unsigned)reader.result);
        ++failures;
    }

    /* Readers run together: four of them hold the lock at the same time. */
    struct Shared shared = {.handle = handle, .lock = PTHREAD_MUTEX_INITIALIZER};
    pthread_t readers[4];
    for (int i = 0; i < 4; ++i) {
        pthread_create(&readers[i], NULL, shared_read_thread, &shared);
    }
    usleep(1000);
    pthread_mutex_lock(&shared.lock);
    const int concurrent = shared.readers;
    pthread_mutex_unlock(&shared.lock);
    for (int i = 0; i < 4; ++i) {
        pthread_join(readers[i], NULL);
    }
    if (concurrent < 2) {
        fprintf(stderr, "FAIL: only %d of 4 readers held the lock together\n", concurrent);
        ++failures;
    }

    /* Timed read with a deadline in the past reports ETIMEDOUT(60). The writer holds the lock on
     * another thread: the same thread would report EDEADLK first. */
    struct WriterArgs writer = {.handle = handle, .release = 0};
    pthread_create(&thread, NULL, writer_thread, &writer);
    usleep(5000);
    const int32_t writer_result = __atomic_load_n(&writer.result, __ATOMIC_ACQUIRE);
    if (writer_result != 0) {
        fprintf(stderr, "FAIL: writer thread returned %#x\n", (unsigned)writer_result);
        ++failures;
    }
    const GuestTime past = {.seconds = 1, .nanoseconds = 0};
    if (rw_timedread_fn(&handle, &past) != ERR(60)) {
        fprintf(stderr, "FAIL: timed read behind a writer did not report ETIMEDOUT\n");
        ++failures;
    }
    __atomic_store_n(&writer.release, 1, __ATOMIC_RELEASE);
    pthread_join(thread, NULL);
    OK(rw_destroy_fn(&handle));
}

int main(void) {
    SemCreate sem_create = (SemCreate)resolve(runtime_sema_resolve, "188x57JYp0g#p#J");
    sem_wait_fn = (SemWait)resolve(runtime_sema_resolve, "Zxa0VhQVTsk#p#J");
    sem_signal_fn = (SemSignal)resolve(runtime_sema_resolve, "4czppHBiriw#p#J");
    sem_poll_fn = (SemPoll)resolve(runtime_sema_resolve, "12wOHk8ywb0#p#J");
    sem_delete_fn = (SemDelete)resolve(runtime_sema_resolve, "R1Jvn8bSCW8#p#J");
    rw_init_fn = (RwInit)resolve(runtime_rwlock_resolve, "6ULAa0fq4jA#p#J");
    rw_destroy_fn = (RwHandleOp)resolve(runtime_rwlock_resolve, "BB+kb08Tl9A#p#J");
    rw_read_fn = (RwHandleOp)resolve(runtime_rwlock_resolve, "Ox9i0c7L5w0#p#J");
    rw_write_fn = (RwHandleOp)resolve(runtime_rwlock_resolve, "mqdNorrB+gI#p#J");
    rw_tryread_fn = (RwHandleOp)resolve(runtime_rwlock_resolve, "XD3mDeybCnk#p#J");
    rw_trywrite_fn = (RwHandleOp)resolve(runtime_rwlock_resolve, "bIHoZCTomsI#p#J");
    rw_unlock_fn = (RwHandleOp)resolve(runtime_rwlock_resolve, "+L98PIbGttk#p#J");
    rw_timedread_fn = (RwTimedOp)resolve(runtime_rwlock_resolve, "iPtZRWICjrM#p#J");
    if (failures) {
        return 1;
    }
    test_semaphores(sem_create);
    test_rwlocks();
    if (failures) {
        printf("runtime sync test FAILED (%d)\n", failures);
        return 1;
    }
    printf("runtime sync test passed\n");
    return 0;
}
