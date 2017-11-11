#include <opencv2/opencv.hpp>

#ifdef _WIN32

#define OCV_VER_STR CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION)

#ifdef _DEBUG
#define OCV_LIB_EXT "d.lib"
#else
#define OCV_LIB_EXT ".lib"
#endif

#endif

#pragma comment( lib, "opencv_core" OCV_VER_STR OCV_LIB_EXT )

int main( int argc, char **argv )
{
    const cv::String keys =
        "{help h usage |     | print this message }"
        "{@int_val     |     | integer value }"
        "{@float_val   |     |floating point value }"
        "{opt          | 1.0 | floating value option } "
        ;
    cv::CommandLineParser parser( argc, argv, keys );


    parser.about( argv[0] );

    if( parser.has( "help" ) || !parser.has( "@int_val" ) )
    {
        parser.printMessage();
        return 1;
    }

    printf( "int value = %d\n", parser.get<int>( 0 ) );

    if( parser.has( "@float_val" ) )
    {
        printf( "float value = %f\n", parser.get<double>( 1 ) );
    }

    printf( "float option = %f\n", parser.get<double>( "opt" ) );

    if( !parser.check() )
    {
        parser.printErrors();
        return 1;
    }

    return 0;
}
