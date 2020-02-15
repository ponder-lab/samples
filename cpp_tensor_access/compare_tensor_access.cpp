#include <time.h>
#include <boost/multi_array.hpp>

#define DIM 100
#define IX3(i, j, k, DIM) ((i)*(DIM)*(DIM) + (j)*(DIM) + (k))

#if 0
template<typename T>
class my_tensor3
{
public:
    my_tensor3( int *dims )
    {
        int data_size = 1;
        for( int i = 0; i < 3; i++ ) { data_size *= dims[i]; }
        data = new T[data_size];

        stride[0] = dims[1] * dims[2];
        stride[1] = dims[2];
    }

    ~my_tensor3()
    {
        delete[] data;
    }

    T &at( int i, int j, int k )
    {
        return data[i*stride[0] + j*stride[1] + k];
    }

private:
    int stride[2];
    T *data;
};

#else

template<typename T, int ORDER>
class my_tensor
{
public:
    my_tensor( int *dims_ )
    {
        int data_size = 1;
        for( int i = 0; i < ORDER; i++ ) { data_size *= dims_[i]; }
        data = new T[data_size];

        stride = new int[ORDER - 1];
        for( int i = 0; i < (ORDER - 1); i++ )
        {
            stride[i] = 1;
            for( int j = (i + 1); j < ORDER; j++ )
            {
                stride[i] *= dims_[j];
            }
        }

        dims = new int[ORDER];
        for( int i = 0; i < ORDER; i++ ) { dims[i] = dims_[i] }
    }

    ~my_tensor()
    {
        delete[] data;
        delete[] stride;
    }

    T &at3( int i, int j, int k )
    {
        return data[i*stride[0] + j*stride[1] + k];
    }

    int order()
    {
        return ORDER;
    }

    int dim( int i )
    {
        return dims[i];
    }
        
private:
    T *data;
    int *stride;
    int *dims;
};
#endif

int main()
{
    double poly_data[DIM*DIM*DIM] = { 0.0 };
    boost::multi_array_ref<double, 3> poly(poly_data, boost::extents[DIM][DIM][DIM]);

    int dims[] = { DIM, DIM, DIM };
    //my_tensor3<double> poly3( dims );
    my_tensor<double, 3> poly3( dims );

    struct timespec start, end0, end1, end2;

    clock_gettime(CLOCK_REALTIME, &start);

    double tmp;

    for( int l = 0; l < 100; l++ )
    {
        tmp = 0.0;
        for( int i = 0; i < DIM; i++ )
        {
            for( int j = 0; j < DIM; j++ )
            {
                for( int k = 0; k < DIM; k++ )
                {
                    poly_data[IX3(i,j,k,DIM)] = poly_data[IX3(i,j,k,DIM)] + tmp;
                    tmp += 1.0;
                }
            }
        }
    }

    clock_gettime(CLOCK_REALTIME, &end0);

    for( int l = 0; l < 100; l++ )
    {
        tmp = 0.0;
        for( int i = 0; i < DIM; i++ )
        {
            for( int j = 0; j < DIM; j++ )
            {
                for( int k = 0; k < DIM; k++ )
                {
                    poly[i][j][k] = poly[i][j][k] + tmp;
                    tmp += 1.0;
                }
            }
        }
    }

    clock_gettime(CLOCK_REALTIME, &end1);

    for( int l = 0; l < 100; l++ )
    {
        tmp = 0.0;
        for( int i = 0; i < DIM; i++ )
        {
            for( int j = 0; j < DIM; j++ )
            {
                for( int k = 0; k < DIM; k++ )
                {
                    //poly3.at(i, j, k) = poly3.at(i, j, k) + tmp;
                    poly3.at3(i, j, k) = poly3.at3(i, j, k) + tmp;
                    tmp += 1.0;
                }
            }
        }
    }

    clock_gettime(CLOCK_REALTIME, &end2);

    if( end0.tv_nsec < start.tv_nsec ) {
        printf("%10ld.%09ld", end0.tv_sec - start.tv_sec - 1
               ,end0.tv_nsec + 1000000000 - start.tv_nsec);
    } else {
        printf("%10ld.%09ld", end0.tv_sec - start.tv_sec
               ,end0.tv_nsec - start.tv_nsec);
    }
    printf("\n");

    if( end1.tv_nsec < end0.tv_nsec ) {
        printf("%10ld.%09ld", end1.tv_sec - end0.tv_sec - 1
               ,end1.tv_nsec + 1000000000 - end0.tv_nsec);
    } else {
        printf("%10ld.%09ld", end1.tv_sec - end0.tv_sec
               ,end1.tv_nsec - end0.tv_nsec);
    }
    printf("\n");

    if( end2.tv_nsec < end1.tv_nsec ) {
        printf("%10ld.%09ld", end2.tv_sec - end1.tv_sec - 1
               ,end2.tv_nsec + 1000000000 - end1.tv_nsec);
    } else {
        printf("%10ld.%09ld", end2.tv_sec - end1.tv_sec
               ,end2.tv_nsec - end1.tv_nsec);
    }
    printf("\n");

    return 0;
}
