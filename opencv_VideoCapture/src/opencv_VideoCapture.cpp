#include <opencv2/opencv.hpp>

int main()
{
    cv::VideoCapture cap;

    if( !cap.open( 0 ) )
    {
        fprintf( stderr, "Failed to open camera\n" );
        return 1;
    }

    while( 1 )
    {
        cv::Mat img_in;
        if( !cap.read( img_in ) )
        {
            break;
        }

        cv::imshow( "img_in", img_in );

        if( cv::waitKey( 1 ) == 27 )
        {
            break;
        }
    }
    
    return 0;
}
