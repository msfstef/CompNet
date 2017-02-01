#include <stdlib.h>
#include <string.h> 

void relax(int L, double p, int* hsd, int* z, int* zc)
{
	int numFalling;
	int numFalling_temp;
	int arr_size = 2*L;
    int indFalling[arr_size];
	int indFalling_temp[arr_size];
	numFalling = 1;
	indFalling[0] = 0;
	int i, n;
    while (numFalling) {
		numFalling_temp = 0;
		memset(indFalling_temp, 0, arr_size*sizeof( int ) );
        for (n = 0; n < numFalling; n++){
			i = indFalling[n];
            if (z[i] <= zc[i]) {continue;}
              
            hsd[1]++ ;
             
            if (i == 0){
                hsd[0]-- ;
                z[i] = z[i] - 2 ;
                z[i+1]++ ;
                if (z[i+1] > zc[i+1]){
                    indFalling_temp[numFalling_temp] = i+1;
					numFalling_temp++ ;
				}
			}
			else if (i == L-1){
                z[i]-- ;
                z[i-1]++ ;
                hsd[2]++ ;
                if (z[i-1] > zc[i-1]){
                    indFalling_temp[numFalling_temp] = i-1;
					numFalling_temp++ ;
				}
            }
            else {
                z[i] = z[i] - 2;
                z[i-1]++ ;
                z[i+1]++ ;
                if (z[i-1] > zc[i-1]){
                    indFalling_temp[numFalling_temp] = i-1;
					numFalling_temp++ ;
				}
                if (z[i+1] > zc[i+1]){
                    indFalling_temp[numFalling_temp] = i+1;
					numFalling_temp++ ;
				}
			}
			double r = ((double)rand()/(double)(RAND_MAX));
            zc[i] = ( r < p ) ? ( 1 ) : ( 2 );
			if (z[i] > zc[i]) {
				indFalling_temp[numFalling_temp] = i;
				numFalling_temp++ ;
			}
		}
		memcpy(indFalling, indFalling_temp, numFalling_temp*sizeof(int));
        numFalling = numFalling_temp;
	}
	return;
}