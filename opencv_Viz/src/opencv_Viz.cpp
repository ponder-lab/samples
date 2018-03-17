#include <opencv2/opencv.hpp>

int main()
{
    /* creating Viz window */
    cv::viz::Viz3d window1( "hello1" );

    /* adding coordinate system */
    window1.showWidget( "coord1", cv::viz::WCoordinateSystem() );

    /* adding a line */
    cv::viz::WLine line( cv::Point3f( -1, -1, -1 ), cv::Point3f( 1, 1, 1 ) );
    line.setRenderingProperty( cv::viz::LINE_WIDTH, 4.0 );
    window1.showWidget( "line", line );

    /* render window contents */
    window1.spin();

    /* creating Viz window in a different way */
    cv::Ptr<cv::viz::Viz3d> window2 = new cv::viz::Viz3d( "hello2" );

    /* adding coordinate system */
    window2->showWidget( "coord2", cv::viz::WCoordinateSystem() );

    /* adding a point cloud */
    cv::Mat pts( 4, 1, CV_32FC3 );
    pts.at<cv::Point3f>( 0 ) = cv::Point3f( 5, 5, 5 );
    pts.at<cv::Point3f>( 1 ) = cv::Point3f( 4, 4, 4 );
    pts.at<cv::Point3f>( 2 ) = cv::Point3f( 3, 4, 4 );
    pts.at<cv::Point3f>( 3 ) = cv::Point3f( 2, 4, 4 );
    cv::viz::WCloud cloud( pts );
    window2->showWidget( "cloud", cloud );

    /* render and wait for 1ms until stopped (pressing 'q') */
    window2->spinOnce( 1, true );
    while( !window2->wasStopped() )
    {
        window2->spinOnce( 1, true );
    }

    return 0;
}
