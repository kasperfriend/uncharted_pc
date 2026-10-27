/* Benchmarks for the guest runtime's synchronization and memory-tracking fast paths.
 *
 * Each block compares the shape the runtime code used before with the shape it uses now, doing
 * the same work on both sides, so the numbers stay meaningful on any host:
 *   - registry lookup: linked-list scan (once per guest rwlock/semaphore operation) against a
 *     hashed bucket (src/runtime_rwlock.c, src/runtime_sema.c),
 *   - blocking wait: the pthread condition variable that a wait no longer creates and destroys
 *     around every wait (src/runtime_sema.c keeps one per thread instead),
 *   - region cache: 8-entry linear probe against page-indexed direct-mapped slots
 *     (src/runtime_memory.c; the cache serves every GPU-side guest range query).
 *
 * Self-contained on purpose: it links nothing from the runtime, so it builds and runs anywhere.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static double best_of(int runs, double (*fn)(void)) {
    double best = 1e18;
    for (int i = 0; i < runs; ++i) {
        const double value = fn();
        if (value < best) best = value;
    }
    return best;
}

/* --- registry lookup: linked-list scan against hashed bucket ------------------------------ */

#define MAX_NODES 128
typedef struct Node {
    uintptr_t handle;
    struct Node *next, *hash_next;
} Node;

static Node nodes[MAX_NODES];
static Node *list_head;
static Node *buckets[256];

static unsigned bucket_of(uintptr_t handle) {
    /* Handles are 16-byte aligned host pointers: drop the low bits, then mix the page bits. */
    uint64_t h = (uint64_t)handle >> 4;
    h ^= h >> 11;
    return (unsigned)(h & 255u);
}

static void registry_add(Node *node) {
    node->next = list_head;
    list_head = node;
    const unsigned bucket = bucket_of(node->handle);
    node->hash_next = buckets[bucket];
    buckets[bucket] = node;
}

static void registry_reset(int count) {
    list_head = NULL;
    memset(buckets, 0, sizeof(buckets));
    for (int i = 0; i < count; ++i) {
        nodes[i].handle = 0x10000000u + (uintptr_t)i * 0x40u;
        registry_add(&nodes[i]);
    }
}

static Node *find_list(uintptr_t handle) {
    for (Node *node = list_head; node; node = node->next) {
        if (node->handle == handle) return node;
    }
    return NULL;
}

static Node *find_hash(uintptr_t handle) {
    for (Node *node = buckets[bucket_of(handle)]; node; node = node->hash_next) {
        if (node->handle == handle) return node;
    }
    return NULL;
}

static uintptr_t probe_handle;
static long lookup_iterations = 4000000;

static double bench_lookup_list(void) {
    const double start = now_ns();
    uintptr_t sink = 0;
    for (long i = 0; i < lookup_iterations; ++i) {
        sink += (uintptr_t)find_list(probe_handle);
    }
    if (!sink) putchar(0);
    return (now_ns() - start) / (double)lookup_iterations;
}

static double bench_lookup_hash(void) {
    const double start = now_ns();
    uintptr_t sink = 0;
    for (long i = 0; i < lookup_iterations; ++i) {
        sink += (uintptr_t)find_hash(probe_handle);
    }
    if (!sink) putchar(0);
    return (now_ns() - start) / (double)lookup_iterations;
}

/* --- blocking wait: condition variable set up per wait ------------------------------------ */

static double bench_cond_setup(void) {
    const long n = 200000;
    const double start = now_ns();
    for (long i = 0; i < n; ++i) {
        pthread_cond_t cond;
        pthread_condattr_t attr;
        pthread_condattr_init(&attr);
        pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
        pthread_cond_init(&cond, &attr);
        pthread_condattr_destroy(&attr);
        pthread_cond_destroy(&cond);
    }
    return (now_ns() - start) / (double)n;
}

/* Cost of an actual wakeup, so the setup cost above can be put in context: two threads hand a
 * token back and forth over a condition variable initialized once. */
static pthread_mutex_t ping_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ping_cond = PTHREAD_COND_INITIALIZER;
static int ping_turn;
static long ping_rounds;

static void *ping_peer(void *arg) {
    (void)arg;
    pthread_mutex_lock(&ping_lock);
    for (long i = 0; i < ping_rounds; ++i) {
        while (ping_turn != 1) pthread_cond_wait(&ping_cond, &ping_lock);
        ping_turn = 0;
        pthread_cond_signal(&ping_cond);
    }
    pthread_mutex_unlock(&ping_lock);
    return NULL;
}

static double bench_ping_pong(void) {
    pthread_t peer;
    ping_rounds = 20000;
    ping_turn = 0;
    const double start = now_ns();
    pthread_create(&peer, NULL, ping_peer, NULL);
    pthread_mutex_lock(&ping_lock);
    for (long i = 0; i < ping_rounds; ++i) {
        ping_turn = 1;
        pthread_cond_signal(&ping_cond);
        while (ping_turn != 0) pthread_cond_wait(&ping_cond, &ping_lock);
    }
    pthread_mutex_unlock(&ping_lock);
    pthread_join(peer, NULL);
    return (now_ns() - start) / (double)ping_rounds;
}

/* --- region cache: 8-entry linear probe against page-indexed direct mapped ---------------- */

typedef struct {
    uint64_t generation;
    uintptr_t start, end;
    int kind;
} CachedRegion;

static uint64_t table_generation = 2;
static __thread CachedRegion region_linear_cache[8];
static __thread unsigned region_linear_next;

static const CachedRegion *cached_region_linear(uintptr_t address) {
    const uint64_t generation = __atomic_load_n(&table_generation, __ATOMIC_ACQUIRE);
    if (generation & 1) return NULL;
    for (unsigned i = 0; i < 8; ++i) {
        const CachedRegion *c = &region_linear_cache[i];
        if (c->generation == generation && c->end && c->start <= address && address < c->end) {
            return c;
        }
    }
    return NULL;
}

static void cache_region_linear(uintptr_t start, uintptr_t end) {
    const uint64_t generation = __atomic_load_n(&table_generation, __ATOMIC_ACQUIRE);
    if (generation & 1) return;
    region_linear_cache[region_linear_next++ % 8] = (CachedRegion){generation, start, end, 1};
}

#define REGION_SLOTS 32
static __thread CachedRegion region_direct_cache[REGION_SLOTS];

static unsigned region_slot(uintptr_t address) {
    uint64_t page = address >> 12;
    page ^= page >> 5;
    return (unsigned)(page & (REGION_SLOTS - 1));
}

static const CachedRegion *cached_region_direct(uintptr_t address) {
    const uint64_t generation = __atomic_load_n(&table_generation, __ATOMIC_ACQUIRE);
    if (generation & 1) return NULL;
    const CachedRegion *c = &region_direct_cache[region_slot(address)];
    if (c->generation == generation && c->start <= address && address < c->end) return c;
    return NULL;
}

static void cache_region_direct(uintptr_t start, uintptr_t end) {
    const uint64_t generation = __atomic_load_n(&table_generation, __ATOMIC_ACQUIRE);
    if (generation & 1) return;
    region_direct_cache[region_slot(start)] = (CachedRegion){generation, start, end, 1};
}

#define REGION_COUNT 32
static uintptr_t region_addresses[REGION_COUNT];
static long region_iterations = 4000000;

static double bench_region_linear(void) {
    for (int i = 0; i < 8; ++i) {
        const uintptr_t base = region_addresses[i] & ~(uintptr_t)0xFFFFFu;
        cache_region_linear(base, base + 0x200000u);
    }
    const double start = now_ns();
    unsigned sink = 0;
    for (long i = 0; i < region_iterations; ++i) {
        sink += cached_region_linear(region_addresses[i & (REGION_COUNT - 1)]) != NULL;
    }
    if (!sink) putchar(0);
    return (now_ns() - start) / (double)region_iterations;
}

static double bench_region_direct(void) {
    for (int i = 0; i < 8; ++i) {
        const uintptr_t base = region_addresses[i] & ~(uintptr_t)0xFFFFFu;
        cache_region_direct(base, base + 0x200000u);
    }
    const double start = now_ns();
    unsigned sink = 0;
    for (long i = 0; i < region_iterations; ++i) {
        sink += cached_region_direct(region_addresses[i & (REGION_COUNT - 1)]) != NULL;
    }
    if (!sink) putchar(0);
    return (now_ns() - start) / (double)region_iterations;
}

int main(void) {
    for (int i = 0; i < REGION_COUNT; ++i) {
        region_addresses[i] =
            0x10000000u + (uintptr_t)i * 0x200000u + 0x1000u * (uintptr_t)(i % 7);
    }

    printf("Registry lookup, per guest rwlock/semaphore operation:\n");
    const int counts[] = {4, 16, 64, 128};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); ++c) {
        registry_reset(counts[c]);
        probe_handle = nodes[counts[c] / 2].handle;
        lookup_iterations = counts[c] >= 64 ? 1000000 : 4000000;
        const double list = best_of(3, bench_lookup_list);
        const double hash = best_of(3, bench_lookup_hash);
        printf("  %3d live objects: list %6.2f ns, hashed %5.2f ns (%4.1fx)\n", counts[c], list,
               hash, list / hash);
    }

    printf("Blocking wait:\n");
    printf("  condition variable created+destroyed per wait: %6.1f ns\n",
           best_of(5, bench_cond_setup));
    printf("  wakeup handoff between two threads:            %6.1f ns\n",
           best_of(3, bench_ping_pong));

    printf("Region cache probe, per guest range query:\n");
    const double linear = best_of(3, bench_region_linear);
    const double direct = best_of(3, bench_region_direct);
    printf("  8-entry linear %6.2f ns, 32-entry direct-mapped %5.2f ns (%4.1fx)\n", linear, direct,
           linear / direct);
    return 0;
}
