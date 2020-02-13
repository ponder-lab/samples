#include <cstdint>
#include <stdio.h>

int main()
{
    FILE *fp;

    /*
      Read data
     */
    fp = fopen( "idata.bin", "rb" );
    if( fp == NULL )
    {
        fprintf( stderr, "Failed to open file (data.bin)\n" );
        return 1;
    }

    int32_t ndim;
    fread( &ndim, 4, 1, fp );

    int32_t *shape = new int32_t[ndim];
    fread( shape, 4, ndim, fp );

    int num_data = 1;
    for( int i = 0; i < ndim; i++ )
    {
        num_data *= shape[i];
    }
    
    float *data = new float[num_data];
    fread( data, 4, num_data, fp );
    
    fclose( fp );

    /*
      Do some processing
     */
    for( int i = 0; i < num_data; i++ )
    {
        data[i] += 1.0;
    }

    /*
      Write data
     */

    fp = fopen( "odata.bin", "wb" );
    if( fp == NULL )
    {
        fprintf( stderr, "Failed to open file (data.bin)\n" );
        return 1;
    }

    fwrite( &ndim, 4, 1, fp );
    fwrite( shape, 4, ndim, fp );
    fwrite( data, 4, num_data, fp );

    delete[] data;
    delete[] shape;

    return 0;
}
