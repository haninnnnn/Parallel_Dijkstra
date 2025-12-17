#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <time.h>

#define INF INT_MAX
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

typedef struct {
    int **A;
    int n;
} Graph;

Graph* make_graph(int n, double d, unsigned int seed) {
    srand(seed);
    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->n = n;
    g->A = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        g->A[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            if (i == j) g->A[i][j] = 0;
            else if ((double)rand() / RAND_MAX < d) g->A[i][j] = rand() % 100 + 1;
            else g->A[i][j] = INF;
        }
    }
    return g;
}

void free_graph(Graph* g) {
    for (int i = 0; i < g->n; i++) free(g->A[i]);
    free(g->A);
    free(g);
}

void dseq(Graph* g, int s, int* dist) {
    int n = g->n;
    int *vis = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) dist[i] = INF;
    dist[s] = 0;

    for (int it = 0; it < n - 1; it++) {
        int u = -1, md = INF;
        for (int v = 0; v < n; v++) {
            if (!vis[v] && dist[v] < md) {
                md = dist[v];
                u = v;
            }
        }
        if (u == -1) break;
        vis[u] = 1;

        for (int v = 0; v < n; v++) {
            if (!vis[v] && g->A[u][v] != INF && dist[u] != INF) {
                int nd = dist[u] + g->A[u][v];
                if (nd < dist[v]) dist[v] = nd;
            }
        }
    }
    free(vis);
}

// -------------------------
// HIGHLY OPTIMIZED KERNELS
// -------------------------

__global__ void init_kernel(int *dist, int n, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dist[idx] = (idx == s) ? 0 : INF;
    }
}

// Optimized: Each thread handles one vertex, checking all incoming edges
__global__ void relax_by_vertex(const int *graph, const int *dist_old, int *dist_new, int *changed, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < n) {
        int min_dist = dist_old[v];

        // Check all incoming edges to vertex v
        for (int u = 0; u < n; u++) {
            int w = graph[u * n + v];
            if (w != INF && dist_old[u] != INF) {
                int new_dist = dist_old[u] + w;
                if (new_dist > dist_old[u] && new_dist < min_dist) {
                    min_dist = new_dist;
                }
            }
        }

        dist_new[v] = min_dist;
        if (min_dist < dist_old[v]) {
            *changed = 1;
        }
    }
}

// Optimized version without shared memory complexity
__global__ void relax_by_vertex_optimized(const int *graph, const int *dist_old, int *dist_new, int *changed, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < n) {
        int min_dist = dist_old[v];
        bool updated = false;

        // Check all incoming edges to vertex v
        for (int u = 0; u < n; u++) {
            int w = graph[u * n + v];
            if (w != INF && dist_old[u] != INF) {
                // Check for valid addition (overflow protection)
                if (dist_old[u] <= INF - w) {
                    int new_dist = dist_old[u] + w;
                    if (new_dist < min_dist) {
                        min_dist = new_dist;
                        updated = true;
                    }
                }
            }
        }

        dist_new[v] = min_dist;
        if (updated) {
            *changed = 1;
        }
    }
}

void dcuda_optimized(Graph* g, int s, int *dist) {
    int n = g->n;
    int total_edges = n * n;

    // Flatten graph
    int *h_graph = (int*)malloc(total_edges * sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_graph[i * n + j] = g->A[i][j];
        }
    }

    int *d_graph, *d_dist_old, *d_dist_new, *d_changed;

    CUDA_CHECK(cudaMalloc(&d_graph, total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist_old, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist_new, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_graph, h_graph, total_edges * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize distances
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<<<blocks, BLOCK_SIZE>>>(d_dist_old, n, s);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Use shared memory version for better performance
    int h_changed;
    int iter = 0;
    int max_iter = n - 1;

    while (iter < max_iter) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        relax_by_vertex_optimized<<<blocks, BLOCK_SIZE>>>(d_graph, d_dist_old, d_dist_new, d_changed, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap buffers
        int *temp = d_dist_old;
        d_dist_old = d_dist_new;
        d_dist_new = temp;

        iter++;

        // Early exit if no changes
        if (!h_changed) break;
    }

    // Copy result back
    CUDA_CHECK(cudaMemcpy(dist, d_dist_old, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_graph);
    cudaFree(d_dist_old);
    cudaFree(d_dist_new);
    cudaFree(d_changed);
    free(h_graph);
}

int verify(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    int n = 1000;
    double dens = 0.3;
    if (argc > 1) n = atoi(argv[1]);

    Graph* g = make_graph(n, dens, 1234 + n);
    int *ds = (int*)malloc(n * sizeof(int));
    int *dc = (int*)malloc(n * sizeof(int));

    clock_t t0 = clock();
    dseq(g, 0, ds);
    clock_t t1 = clock();
    double ts = (double)(t1 - t0) / CLOCKS_PER_SEC;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    dcuda_optimized(g, 0, dc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double tp = ms / 1000.0;

    double sp = ts / tp;
    double eff = sp * 100.0;
    int ok = verify(ds, dc, n);

    printf("Algorithm,CUDA_Optimized,Graph_Size,%d,Seq_Time,%.6f,Par_Time,%.6f,Speedup,%.4f,Efficiency,%.2f,Correct,%d\n",
           n, ts, tp, sp, eff, ok);

    free(ds);
    free(dc);
    free_graph(g);
    return 0;
}
