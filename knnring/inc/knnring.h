#ifndef KNNRING_H
#define KNNRING_H
typedef struct knnresult{
  int * nidx;
  double * ndist;
  int m;
  int k;
}knnresult;

knnresult kNN(double * X,double * Y,int n,int m,int d,int k);
knnresult distrAllkNN(double * X,int n,int d,int k);
#endif
