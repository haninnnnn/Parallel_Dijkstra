#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <time.h>
#define INF INT_MAX
typedef struct { int **A; int n; } G;

G* make_graph(int n, double d, unsigned long long seed){
    srand(seed);
    G* g=(G*)malloc(sizeof(G)); g->n=n; g->A=(int**)malloc(n*sizeof(int*));
    for(int i=0;i<n;i++){ g->A[i]=(int*)malloc(n*sizeof(int));
        for(int j=0;j<n;j++){
            if(i==j) g->A[i][j]=0;
            else if ((double)rand()/RAND_MAX<d) g->A[i][j]=rand()%100+1;
            else g->A[i][j]=INF;
        }
    } return g;
}
void free_graph(G* g){ for(int i=0;i<g->n;i++) free(g->A[i]); free(g->A); free(g); }
void dseq(G* g,int s,int*dist){ int n=g->n; int*vis=(int*)calloc(n,sizeof(int));
    for(int i=0;i<n;i++) dist[i]=INF; dist[s]=0;
    for(int it=0; it<n-1; it++){
        int u=-1, md=INF;
        for(int v=0; v<n; v++) if(!vis[v] && dist[v]<md){ md=dist[v]; u=v; }
        if(u==-1) break; vis[u]=1;
        for(int v=0; v<n; v++){
            if(!vis[v] && g->A[u][v]!=INF && dist[u]!=INF){
                int nd=dist[u]+g->A[u][v]; if(nd<dist[v]) dist[v]=nd;
            }
        }
    } free(vis);
}
void dome(G* g,int s,int*dist,int th){ int n=g->n; int*vis=(int*)calloc(n,sizeof(int));
    for(int i=0;i<n;i++) dist[i]=INF; dist[s]=0; omp_set_num_threads(th);
    for(int it=0; it<n-1; it++){
        int u=-1, md=INF;
        #pragma omp parallel
        {
            int lu=-1, lm=INF;
            #pragma omp for nowait
            for(int v=0; v<n; v++) if(!vis[v] && dist[v]<lm){ lm=dist[v]; lu=v; }
            #pragma omp critical
            { if(lm<md){ md=lm; u=lu; } }
        }
        if(u==-1) break; vis[u]=1;
        #pragma omp parallel for schedule(static)
        for(int v=0; v<n; v++){
            if(!vis[v] && g->A[u][v]!=INF && dist[u]!=INF){
                int nd=dist[u]+g->A[u][v]; if(nd<dist[v]) dist[v]=nd;
            }
        }
    } free(vis);
}
int verify(int*a,int*b,int n){ for(int i=0;i<n;i++) if(a[i]!=b[i]) return 0; return 1; }
int main(){
    int sizes[]={500,1000,2000}; int ns=3; double dens=0.3; int ths[]={1,2,4,8}; int nt=4;
    printf("graph_size,density,threads,seq_time,par_time,speedup,efficiency,correct\n");
    for(int i=0;i<ns;i++){
        int n=sizes[i]; G* g=make_graph(n,dens,1234+n);
        int* ds=(int*)malloc(n*sizeof(int)); double t0=omp_get_wtime(); dseq(g,0,ds); double ts=omp_get_wtime()-t0;
        for(int j=0;j<nt;j++){
            int th=ths[j]; int* dp=(int*)malloc(n*sizeof(int));
            t0=omp_get_wtime(); dome(g,0,dp,th); double tp=omp_get_wtime()-t0;
            double sp=ts/tp; double eff=(sp/th)*100.0; int ok=verify(ds,dp,n);
            printf("%d,%.2f,%d,%.6f,%.6f,%.4f,%.2f,%d\n", n,dens,th,ts,tp,sp,eff,ok);
            free(dp);
        } free(ds); free_graph(g);
    } return 0;
}
