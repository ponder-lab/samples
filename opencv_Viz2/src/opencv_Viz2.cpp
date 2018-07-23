#include <opencv2/opencv.hpp>

int main()
{
    cv::viz::Viz3d window_3d( "3D window" );

    window_3d.showWidget( "Coordinate system",
                          cv::viz::WCoordinateSystem() );

    cv::Mat cloud = ( cv::Mat_<float>( 8, 3 ) <<
                       0.0,  0.0,  0.0,
                      10.0,  0.0,  0.0,
                       0.0, 10.0,  0.0,
                      10.0, 10.0,  0.0,
                       0.0,  0.0, 10.0,
                      10.0,  0.0, 10.0,
                       0.0, 10.0, 10.0,
                      10.0, 10.0, 10.0 );
    cloud = cloud.reshape( 3, cloud.rows );
    cv::viz::WCloud cloud_widget( cloud, cv::viz::Color::green() );
    window_3d.showWidget( "Cloud", cloud_widget );

    cv::Mat vec_R = ( cv::Mat_<float>( 3, 1 ) <<
                      30.0/180.0*CV_PI, 0.0, 0.0 );
    cv::Mat mat_R;
    cv::Rodrigues( vec_R, mat_R );
    std::cout << "mat_R=" << mat_R << std::endl;

    /*
      mat_R, vec_t are the Rotation matrix / translation vector
      to transform from camera coordinates to global coordinates
        x_g = R * x_c + t
    */
    cv::Vec3f vec_t( 1.0, 0.0, 0.0 );
    cv::Affine3f cam_pose( mat_R, vec_t );
    
    cv::viz::WCameraPosition cam_coord( 0.5 );
    cv::viz::WCameraPosition frustum( cv::Vec2f( 0.89, 0.52 ) );
    window_3d.showWidget( "Camera coordinate", cam_coord, cam_pose );
    window_3d.showWidget( "Frustum", frustum, cam_pose );

    while( !window_3d.wasStopped() )
    {
        vec_R.at<float>(0) += 5.0 / 180.0 * CV_PI;
        cv::Rodrigues( vec_R, mat_R );
        cv::Affine3f new_cam_pose( mat_R, vec_t );

        window_3d.setWidgetPose( "Camera coordinate", new_cam_pose );
        window_3d.setWidgetPose( "Frustum", new_cam_pose );

        window_3d.removeWidget( "Cloud" );
        cloud += cv::Scalar::all(0.01);
        cv::viz::WCloud new_cloud_widget( cloud, cv::viz::Color::green() );
        window_3d.showWidget( "Cloud", new_cloud_widget );
        
        window_3d.spinOnce( 33, true );
    }
    
    return 0;
}
