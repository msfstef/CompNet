#include <stdlib.h>
#include <string.h> 

void relax(int L, float p, int* hs, int* z, int* zc)
{
	int numFalling;
	int numFalling_temp;
    int indFalling[L];
	int indFalling_temp[L];
	numFalling = 1;
	indFalling[0] = 0;
	int i, n;
    while (numFalling) {
		numFalling_temp = 0;
		memset(indFalling_temp, 0, L*sizeof( int ) );
        for (n = 0; n < numFalling; n++){
			i = indFalling[n];
            if (z[i] <= zc[i]) {continue;}
              
            hs[1]++ ;
             
            if (i == 0){
                hs[0]-- ;
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
			float r = ((float)rand()/(float)(RAND_MAX));
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