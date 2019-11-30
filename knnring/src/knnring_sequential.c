#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <Accelerate/Accelerate.h>
#include <cblas.h>
#include "knnring.h"



void swap(double * arr,int a,int b){
  double temp=arr[a];
  arr[a]=arr[b];
  arr[b]=temp;
}
void swapint(int * arr,int a,int b){
  double temp=arr[a];
  arr[a]=arr[b];
  arr[b]=temp;
}


int partition(double * arr, int * id,int l, int r)
{
    double x = arr[r];
    int i = l;
    for (int j = l; j <= (r - 1); j++) {
        if (arr[j] <= x) {
            swap(arr,i,j);
            swapint(id,i,j);
            i++;
        }
    }
    swap(arr,i,r);
    swapint(id,i,r);
    return i;
}
void QSort(double * arr,int * id, int l, int r){
  int p;
  if(l<r){
    p=partition(arr,id,l,r);
    QSort(arr,id,l,p-1);
    QSort(arr,id,p+1,r);
  }
}

double QS(double * arr,int * id, int l, int r, int k)
{
    // If k is smaller than number of
    // elements in array
    if ( (k > 0) && k <=( r - l + 1) ) {

        // Partition the array around last
        // element and get position of pivot
        // element in sorted array
        int index = partition(arr,id, l, r);

        // If position is same as k
        if ((index - l) == (k - 1))
            return arr[index];

        // If position is more, recur
        // for left subarray
        if ((index - l )> (k - 1))
            return QS(arr,id,l, index - 1, k);

        // Else recur for right subarray
        return QS(arr,id,index + 1, r,k - index + l - 1);
    }

    // If k is more than number of
    // elements in array
    return -100;
}




knnresult kNN(double * X,double * Y,int n,int m,int d,int k){
  knnresult result;
  double dhelper=0;
  result.m=m;
  result.k=k;
  result.nidx=(int *)malloc(m*k*sizeof(int));
  result.ndist=(double *)malloc(m*k*sizeof(double));
  double * distances=(double *)malloc(m*n*sizeof(double));
  int * ids =(int *)malloc(m*n*sizeof(int));
  //BLAS routine to calculate -2*X*Ytrans
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, Y, d, X, d, 0, distances, n);


  //add sumy.^2
  for(int i=0;i<m;i++){
    dhelper=0;
    for(int j=0;j<d;j++){
      dhelper+=Y[i*d+j]*Y[i*d+j];
    }
    for(int j=0;j<n;j++){
      distances[i*n+j]+=dhelper;
    }
  }
  //add sumx.^2
  for(int i=0;i<n;i++){
    dhelper=0;
    for(int j=0;j<d;j++){
      dhelper+=X[i*d+j]*X[i*d+j];
    }
    for(int j=0;j<m;j++){
      distances[j*n+i]+=dhelper;
    }
  }
  //apply sqrt
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      distances[i*n+j]=sqrt(distances[i*n+j]);
    }
  }
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      printf("%f ",distances[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      ids[i*n+j]=j;
    }
  }
  for(int i=0;i<m;i++){
    QS(distances,ids,i*n,(i+1)*n-1,k);

    for(int j=0;j<k;j++){
      result.ndist[i*k+j]=distances[i*k+j];
      result.nidx[i*k+j]=ids[i*k+j];
    }
    QSort(result.ndist,result.nidx,i*k,(i+1)*k-1);
  }


  return result;
}


int main(int argc,char* argv[]){
  knnresult res;
  double X[12];
  int id[12];
  for(int i=0;i<12;i++){
    X[i]=(double)(rand()%1000)/100;
    id[i]=i;
  }
  res=kNN(X,X,12,12,1,7);
  for(int i=0;i<12;i++){
    printf("%f ",X[i]);
  }
  printf("\n");
  for(int i=0;i<res.k;i++){
      printf("%f %d \n",res.ndist[i],res.nidx[i]);
  }

  printf("\n");
  for(int i=0;i<12;i++){
    printf("%d ",id[i]);
  }
  printf("\n%f\n",QS(X,id,0,11,4));
  for(int i=0;i<12;i++){
    printf("%f ",X[i]);
  }
  printf("\n");
  for(int i=0;i<12;i++){
    printf("%d ",id[i]);
  }
  printf("\n");
  return 0;
}
