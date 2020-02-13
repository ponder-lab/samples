#include <iostream>
//#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

double my_add( double a, double b )
{
    return a + b;
}

void show_array( np::ndarray a )
{
    float *data = (float *) a.get_data();
    int ndim = a.get_nd();

    int num_data = 1;
    for( int i = 0; i < ndim; i++ )
    {
        num_data *= a.shape( i );
    }

    for( int i = 0; i < num_data; i++ )
    {
        std::cout << data[i] << ",";
    }
    std::cout << std::endl;
}

void show_list( py::list a )
{
    py::ssize_t len = py::len( a );

    for( int i = 0; i < len; i++ )
    {
        double val = py::extract<double>( a[i] );
        std::cout << val << ",";
    }
    std::cout << std::endl;
}

BOOST_PYTHON_MODULE(my_add)
{
    /* following 2 lines needed for python_numpy */
    Py_Initialize();
    np::initialize();
    
    py::def( "my_add", my_add );
    py::def( "show_array", show_array );
    py::def( "show_list", show_list );
}
