#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#define max(a, b) (a < b? b : a)
#define min(a, b) (a > b? b : a)

double getDistance(char *x, char *y, double icost, double dcost, double rcost){
    unsigned int c = strlen(x);
    unsigned int r = strlen(y);

    double **D = (double**)malloc((r+1) * sizeof(double*));
    for(int i=0; i<=r; i++){
        D[i] = (double*)malloc((c+1) * sizeof(double));
    }

    for(int j=0; j<=r; j++){
        for(int i=0; i<=c; i++){
            D[j][i] = 0.0;
        }
    }

    for(int i=0; i<=c; i++){
        D[0][i] = i;
    }
    for(int j=0; j<=r; j++){
        D[j][0] = j;
    }

    for(int j=1; j<=r; j++){
        for(int i=1; i<=c; i++){
            if(x[i-1] == y[j-1]) D[j][i] = D[j-1][i-1];
            else{
                double replace = D[j-1][i-1] + rcost, insert = D[j-1][i] + icost, remove = D[j][i-1] + dcost;
                D[j][i] = min(replace, min(insert, remove));
            }
        }
    }

    double d = D[r][c];

    for(int i=0; i<=r; i++) free(D[i]);
    free(D);

    return 1 - d / max(r, c);
}

typedef struct{
    char *fileInfo[500];
}Struct;

extern double* getAdjMatrix(Struct *t, int N, double icost, double dcost, double rcost){

    double *adjMatrix = (double *)malloc(N * N * sizeof(double*));

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            char* x = t->fileInfo[i], *y = t->fileInfo[j];
            adjMatrix[i*N + j] = getDistance(x, y, icost, dcost, rcost);
        }
    }

    return adjMatrix;
}