typedef struct knnresult{
  int * nidx;
  double * ndist;
  int m;
  int k;
}knnresult;

knnresult kNN(double * X,double * Y,int n,int m,int d,int k);
