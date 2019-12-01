#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include "knnring.h"
#include "mpi.h"

void swap(double* arr, int a, int b) {
    double temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}
void swapint(int* arr, int a, int b) {
    int temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}

int partition(double* arr, int* id, int l, int r)
{
    double x = arr[r];
    int i = l;
    for (int j = l; j <= (r - 1); j++) {
        if (arr[j] <= x) {
            swap(arr, i, j);
            swapint(id, i, j);
            i++;
        }
    }
    swap(arr, i, r);
    swapint(id, i, r);
    return i;
}
void QSort(double* arr, int* id, int l, int r) {
    int p;
    if (l < r) {
        p = partition(arr, id, l, r);
        QSort(arr, id, l, p - 1);
        QSort(arr, id, p + 1, r);
    }
}
double QS(double* arr, int* id, int l, int r, int k)
{
    // If k is smaller than number of
    // elements in array
    if ((k > 0) && k <= (r - l + 1)) {

        // Partition the array around last
        // element and get position of pivot
        // element in sorted array
        int index = partition(arr, id, l, r);

        // If position is same as k
        if ((index - l) == (k - 1))
            return arr[index];

        // If position is more, recur
        // for left subarray
        if ((index - l) > (k - 1))
            return QS(arr, id, l, index - 1, k);

        // Else recur for right subarray
        return QS(arr, id, index + 1, r, k - index + l - 1);
    }

    // If k is more than number of
    // elements in array
    return -100;
}




knnresult kNN(double* X, double* Y, int n, int m, int d, int k)
{
    knnresult result;
    double dhelper = 0;
    result.m = m;
    result.k = k;
    result.nidx = (int*)malloc(m * k * sizeof(int));
    result.ndist = (double*)malloc(m * k * sizeof(double));
    double* distances = (double*)malloc(m * n * sizeof(double));
    int* ids = (int*)malloc(m * n * sizeof(int));
    //BLAS routine to calculate -2*X*Ytrans
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, Y, d, X, d, 0, distances, n);

    //add sumy.^2
    for (int i = 0; i < m; i++) {
        dhelper = 0;
        for (int j = 0; j < d; j++) {
            dhelper += Y[i * d + j] * Y[i * d + j];
        }
        for (int j = 0; j < n; j++) {
            distances[i * n + j] += dhelper;
        }
    }
    //add sumx.^2
    for (int i = 0; i < n; i++) {
        dhelper = 0;
        for (int j = 0; j < d; j++) {
            dhelper += X[i * d + j] * X[i * d + j];
        }
        for (int j = 0; j < m; j++) {
            distances[j * n + i] += dhelper;
        }
    }
    //apply sqrt
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        {
            distances[i * n + j] = sqrt(distances[i * n + j]);

        }
    }
    printf("\n");

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ids[i * n + j] = j;
        }
    }
    for (int i = 0; i < m; i++)
    {
        QS(distances, ids, i * n, (i + 1) * n - 1, k);

        for (int j = 0; j < k; j++) {
            result.ndist[i * k + j] =  distances[i * n + j];
            result.nidx[i * k + j] = ids[i * n + j];
        }
        QSort(result.ndist, result.nidx, i * k, (i + 1) * k - 1);
    }
    return result;
}


knnresult distrAllkNN(double * X,int n,int d,int k){
  //where new and old results are kept
  knnresult temp,result;
  //helpers for merging temp & result
  int * idhelper=(int *)malloc(k*sizeof(int));
  double * disthelper=(double *)malloc(k*sizeof(double));
  //initializing temp
  temp.m=n;
  temp.k=k;
  temp.nidx=(int *)malloc(n*k*sizeof(int));
  temp.ndist=(double *)malloc(n*k*sizeof(double));
  //initialize result
  result.m=n;
  result.k=k;
  result.nidx=(int *)malloc(n*k*sizeof(int));
  result.ndist=(double *)malloc(n*k*sizeof(double));
  //me = my process id
  //size = num of processes
  //src = where i receive data from
  //dest = to whom i send data
  int me, size,src,dest;
  //we need a temp corpus for all processes that receive while computing
  double * corpus = (double* )malloc(n*d*sizeof(double));
    for(int i=0;i<n*d;i++){
        corpus[i]=X[i];
    }
  double * tempcorpus =(double *)malloc(n*d*sizeof(double));
  //helps to swap tempcorpus and corpus
  double * tempHELP;
  //my resulting id array
  int * ids = (int *)malloc(n*size*sizeof(int));
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status status;
  MPI_request req;
  //i receive from the proc before me in the ring(if im 0 the guy before is size-1)
  if(me==0){
    src=size-1;
  }
  else{
    src=me-1;
  }
  //i send to the next proc in the ring(if im size-1 next is 0)
  if(me==size-1){
    dest=0;
  }
  else{
    dest=me+1;
  }
  //then i communicate
  for(int s=0;s<size-1;s++){
        
    MPI_ISend(corpus,n*d,MPI_DOUBLE,dest,3,MPI_COMM_WORLD,&req);
    MPI_IRecv(tempcorpus,n*d,MPI_DOUBLE,src,3,MPI_COMM_WORLD,&req);
      

    if(i==0){
    result=kNN(corpus,X,n,n,d,k);
    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            if((src-s)==0){
                result.nidx[i*n+j]=result.nidx[i*n+j]+(size-1)*n;
            }
            else{
                result.nidx[i*n+j]=result.nidx[i*n+j]+(src-s-1)*n;
            }
        }
    }
    }
    else{
        temp=kNN(corpus,X,n,n,d,k);
        for(int i=0;i<n;i++){
            for(int j=0;j<k;j++){
                if((src-s)==0){
                    temp.nidx[i*k+j]=temp.nidx[i*k+j]+(size-1)*n;
                }
                else{
                    temp.nidx[i*k+j]=temp.nidx[i*k+j]+(src-s-1)*n;
                }
            }
        }
    }
    //Now combine temp and result and store in result
    if(i>0){
    int i;
    int j;
    for(int ii=0;ii<n;ii++){
    i=0;
    j=0;
        while((i+j)<k){
            if(result.ndist[ii*k+i]<=temp.ndist[ii*k+j]){
            disthelper[i+j]=result.ndist[ii*k+i];
            idhelper[i+j]=result.nidx[ii*k+i];
            i++;
        }
        else{
            disthelper[i+j]=temp.ndist[ii*k+j];
                        idhelper[i+j]=temp.nidx[ii*k+j];
                        j++;

        }
        }
    for(int jj=0;jj<k;jj++){
        result.ndist[ii*k+jj]=disthelper[jj];
        result.nidx[ii*k+jj]=idhelper[jj];
    }
    }
    }
    MPI_Wait(&req,&status);
    MPI_Wait(&req,&status);
    //swap the two buffers so that you do calc on the one you just received each time
    tempHELP=corpus;
    corpus=tempcorpus;
    tempcorpus=tempHELP;
  }
    ///
  result=kNN(X,X,n,n,d,k);
    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            if(me==0){
                result.nidx[i*k+j]=result.nidx[i*k+j]+(size-1)*n;
            }
            else{
                result.nidx[i*k+j]=result.nidx[i*k+j]+(me-1)*n;
            }
        }
    }
  
    return result;
}
