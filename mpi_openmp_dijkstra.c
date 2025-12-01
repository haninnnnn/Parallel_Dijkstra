#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <mpi.h>
#define INF INT_MAX
int** make_graph(int n,double d,unsigned int seed){
    srand(seed); int** A=(int**)malloc(n*sizeof(int*));
    for(int i=0;i<n;i++){ A[i]=(int*)malloc(n*sizeof(int));
        for(int j=0;j<n;j++){
            if(i==j) A[i][j]=0;
            else if((double)rand()/RAND_MAX<d) A[i][j]=rand()%100+1;
            else A[i][j]=INF;
        }
    } return A;
}
void free_graph(int**A,int n){ for(int i=0;i<n;i++) free(A[i]); free(A); }
int main(int argc,char**argv){
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int n=1000; double dens=0.3;
    int** A=NULL;
    if(rank==0) A=make_graph(n,dens,1234);
    // Broadcast matrix row-by-row
    if(rank!=0){ A=(int**)malloc(n*sizeof(int*)); for(int i=0;i<n;i++) A[i]=(int*)malloc(n*sizeof(int)); }
    for(int i=0;i<n;i++) MPI_Bcast(A[i], n, MPI_INT, 0, MPI_COMM_WORLD);

    int* dist=(int*)malloc(n*sizeof(int)); int* vis=(int*)calloc(n,sizeof(int));
    for(int i=0;i<n;i++) dist[i]=INF; dist[0]=0;
    double t0=MPI_Wtime();
    for(int it=0; it<n-1; it++){
        int local_min=INF, local_u=-1;
        for(int v=rank; v<n; v+=size){ if(!vis[v] && dist[v]<local_min){ local_min=dist[v]; local_u=v; } }
        struct { int val; int idx; } in={local_min, local_u}, out;
        MPI_Allreduce(&in,&out,1,MPI_2INT,MPI_MINLOC,MPI_COMM_WORLD);
        int u=out.idx; if(u==-1) break; vis[u]=1;
        #pragma omp parallel for
        for(int v=0; v<n; v++){
            if(!vis[v] && A[u][v]!=INF && dist[u]!=INF){
                int nd=dist[u]+A[u][v]; if(nd<dist[v]) dist[v]=nd;
            }
        }
    }
    double t=MPI_Wtime()-t0;
    if(rank==0) printf("graph_size,%d,mpi_procs,%d,time,%.6f\n", n, size, t);
    free(dist); free(vis); free_graph(A,n);
    MPI_Finalize(); return 0;
}
