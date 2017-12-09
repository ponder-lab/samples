#include <opencv2/opencv.hpp>
#include "opencv2/optflow.hpp"

#ifdef _WIN32

#define OCV_VER_STR CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR) CVAUX_STR(CV_VERSION_REVISION)

#ifdef _DEBUG
#define OCV_LIB_EXT "d.lib"
#else
#define OCV_LIB_EXT ".lib"
#endif

#pragma comment( lib, "opencv_core" OCV_VER_STR OCV_LIB_EXT )
#pragma comment( lib, "opencv_highgui" OCV_VER_STR OCV_LIB_EXT )
#pragma comment( lib, "opencv_videoio" OCV_VER_STR OCV_LIB_EXT )
#pragma comment( lib, "opencv_imgproc" OCV_VER_STR OCV_LIB_EXT )
#pragma comment( lib, "opencv_video" OCV_VER_STR OCV_LIB_EXT )
#pragma comment( lib, "opencv_optflow" OCV_VER_STR OCV_LIB_EXT )

#endif

//#define MOTION_COMPENSATION_DEMO

struct optical_flow_data
{
    cv::Ptr<cv::DenseOpticalFlow> ofp;
    cv::Mat img_prev;

#ifdef MOTION_COMPENSATION_DEMO
    cv::Mat img_pos;
#endif
};

void optical_flow_init( optical_flow_data &of, cv::Size &size_in )
{
    //of.ofp = cv::DualTVL1OpticalFlow::create();
    of.ofp = cv::FarnebackOpticalFlow::create();
    //of.ofp = cv::optflow::createOptFlow_DIS();
    //of.ofp = cv::optflow::createOptFlow_PCAFlow();
    //of.ofp = cv::optflow::createOptFlow_DeepFlow();

#ifdef MOTION_COMPENSATION_DEMO
    of.img_pos.create( size_in, CV_32FC2 );

    for( int y = 0; y < size_in.height; y++ )
    {
        for( int x = 0; x < size_in.width; x++ )
        {
            of.img_pos.at<cv::Point2f>( y, x ) = cv::Point2f( x, y );
        }
    }
#endif
}

void optical_flow_exec( optical_flow_data &of, const cv::Mat &img_in, cv::Mat &img_dbg )
{
    cv::Mat img_curr;
    cv::Mat img_of;
    cv::Mat img_map;


    cv::cvtColor( img_in, img_curr, CV_BGR2GRAY );

    if( of.img_prev.empty() )
    {
        img_in.copyTo( img_dbg );
        img_curr.copyTo( of.img_prev );
        return;
    }

    img_of.create( img_in.size(), CV_32FC2 );
    of.ofp->calc( of.img_prev, img_curr, img_of );

#ifdef MOTION_COMPENSATION_DEMO
    img_map = of.img_pos + img_of;

    cv::remap( img_in, img_dbg, img_map, cv::Mat(), cv::INTER_CUBIC );

#if 0
    cv::Mat img_diff;
    cv::absdiff( of.img_prev, img_dbg, img_diff );
    cv::imshow( "diff", img_diff );
#endif

#else
    img_in.copyTo( img_dbg );
    for( int y = 0; y < img_dbg.rows; y += 8 )
    {
        for( int x = 0; x < img_dbg.cols; x += 8 )
        {
            cv::Point2f pt;
            cv::Point2f vec;
            cv::Point2f end;


            pt = cv::Point2f( x, y );
            vec = img_of.at<cv::Point2f>( y, x );
            end = pt + vec;

            cv::line( img_dbg, pt, end, CV_RGB(0,255,0) );
        }
    }
#endif

    img_curr.copyTo( of.img_prev );
}

int main( int argc, char **argv )
{
    const cv::String keys =
        "{@avi_file   |     | AVI file }"
        ;

    cv::CommandLineParser parser( argc, argv, keys );

    cv::VideoCapture cap;
    optical_flow_data of;
    cv::Size size_in;


    if( parser.has( "@avi_file" ) )
    {
        if( !cap.open( parser.get<cv::String>( 0 ) ) )
        {
            fprintf( stderr, "Failed to open AVI file (%s)\n",
                     parser.get<cv::String>( 0 ).c_str() );
            return 1;
        }
    }
    else
    {
        if( !cap.open( 0 ) )
        {
            fprintf( stderr, "Failed to open camera\n" );
            return 1;
        }
    }

    size_in.width  = (int) cap.get( cv::CAP_PROP_FRAME_WIDTH );
    size_in.height = (int) cap.get( cv::CAP_PROP_FRAME_HEIGHT );

    optical_flow_init( of, size_in );

    while( 1 )
    {
        cv::Mat img_in;
        cv::Mat img_dbg;


        if( !cap.read( img_in ) )
        {
            break;
        }

        optical_flow_exec( of, img_in, img_dbg );

        cv::imshow( "img_in", img_in );
        cv::imshow( "img_dbg", img_dbg );

        if( cv::waitKey( 1 ) == 27 )
        {
            break;
        }
    }

    return 0;
}
