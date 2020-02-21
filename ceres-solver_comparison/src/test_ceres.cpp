#include <ceres/ceres.h>

#define DIM 1000
//#define SQUARE

/*
  Finding intersection of (dim) number of hyper planes

  a(x0 - 1) + b(x1 - 1) + ... + c(xn_2 - 1) + (xn_1 - 1) = 0
  d(x0 - 1) + e(x1 - 1) + ... + f(xn_2 - 1) + (xn_1 - 1) = 0
  ...
  g(x0 - 1) + h(x1 - 1) + ... + i(xn_2 - 1) + (xn_1 - 1) = 0

  should return x0 = x1 = ... = 1
 */

/*
  Cost function and jacobian computation
 */
void cost_jac( int dim, const double *coefs,
               double const* const* x,
               double *res,
               double **jac )
{
    for( int i = 0; i < dim; i++ )
    {
        res[i] = 0.0;
        int j;
        for( j = 0; j < (dim - 1); j++ )
        {
            res[i] += coefs[i*(dim-1)+j] * ( x[0][j] - 1.0 );
        }
        res[i] += x[0][j] - 1.0;
    }

    if( jac != NULL && jac[0] != NULL )
    {
        for( int i = 0; i < dim; i++ )
        {
#ifdef SQUARE
            int j;
            for( j = 0; j < (dim - 1); j++ )
            {
                jac[0][i*dim+j] = 2.0 * res[i] * coefs[i*(dim-1)+j];
            }
            jac[0][i*dim+j] = 2.0 * res[i];
#else
            int j;
            for( j = 0; j < (dim - 1); j++ )
            {
                jac[0][i*dim+j] = coefs[i*(dim-1)+j];
            }
            jac[0][i*dim+j] = 1.0;
#endif
        }
    }

#ifdef SQUARE
    for( int i = 0; i < dim; i++ )
    {
        res[i] *= res[i];
    }
#endif
}

/*
  Cost function only
 */
template <typename T>
void cost( int dim, const double *coefs,
           const T* const x, T *res )
{
    for( int i = 0; i < dim; i++ )
    {
        res[i] = T(0.0);
        int j;
        for( j = 0; j < (dim - 1); j++ )
        {
            res[i] += T(coefs[i*(dim-1)+j]) * ( x[j] - T(1.0) );
        }
        res[i] += x[j] - T(1.0);
#ifdef SQUARE
        res[i] *= res[i];
#endif
    }
}

/*
  Cost function wrapper classes for Ceres-Solver
 */

class analytic_cost_function :
    public ceres::SizedCostFunction<DIM, DIM>
{
public:
    analytic_cost_function( const double *coefs ) :
        m_coefs( coefs )
    {}
    
    virtual ~analytic_cost_function() {}
    virtual bool Evaluate( double const* const* x,
                           double *res,
                           double **jac ) const
    {
        cost_jac( DIM, m_coefs, x, res, jac );
        return true;
    }

    const double *m_coefs;
};

class dynamic_analytic_cost_function :
    public ceres::CostFunction
{
public:
    dynamic_analytic_cost_function( int dim, const double *coefs ) :
        m_dim( dim ),
        m_coefs( coefs )
    {
        set_num_residuals( dim );
        *mutable_parameter_block_sizes() = std::vector<int32_t>{ dim };
    }
    
    virtual ~dynamic_analytic_cost_function() {}
    virtual bool Evaluate( double const* const* x,
                           double *res,
                           double **jac ) const
    {
        cost_jac( m_dim, m_coefs, x, res, jac );
        return true;
    }

    int m_dim;
    const double *m_coefs;
};

struct cost_functor_autodiff
{
    cost_functor_autodiff( const double *coefs ) :
        m_coefs( coefs )
    {}

    template <typename T>
    bool operator()( const T* const x, T *res ) const
    {
        cost<T>( DIM, m_coefs, x, res );
        return true;
    }

    const double *m_coefs;
};

struct cost_functor_dynamic_autodiff
{
    cost_functor_dynamic_autodiff( int dim, const double *coefs ) :
        m_dim( dim ),
        m_coefs( coefs )
    {}

    template <typename T>
    bool operator()(T const* const* x, T* res) const
    {
        cost<T>( m_dim, m_coefs, x[0], res );
        return true;
    }

    int m_dim;
    const double *m_coefs;
};

struct cost_functor_numeric
{
    cost_functor_numeric( const double *coefs ) :
        m_coefs( coefs )
    {}

    bool operator()( const double * const x, double *res ) const
    {
        cost<double>( DIM, m_coefs, x, res );
        return true;
    }

    const double *m_coefs;
};

struct cost_functor_dynamic_numeric
{
    cost_functor_dynamic_numeric( int dim, const double *coefs ) :
        m_dim( dim ),
        m_coefs( coefs )
    {}

    bool operator()( double const* const* x, double *res ) const
    {
        cost<double>( m_dim, m_coefs, x[0], res );
        return true;
    }

    int m_dim;
    const double *m_coefs;
};

ceres::CostFunction *gen_dynamic_autodiff_cost_function( int dim, const double *coefs )
{
    ceres::DynamicAutoDiffCostFunction<cost_functor_dynamic_autodiff>* cost_function =
        new ceres::DynamicAutoDiffCostFunction<cost_functor_dynamic_autodiff>(
            new cost_functor_dynamic_autodiff( dim, coefs ) );
    cost_function->AddParameterBlock( dim );
    cost_function->SetNumResiduals( dim );
    return cost_function;
}

ceres::CostFunction *gen_dynamic_numeric_cost_function( int dim, const double *coefs )
{
    ceres::DynamicNumericDiffCostFunction<cost_functor_dynamic_numeric> *cost_function =
        new ceres::DynamicNumericDiffCostFunction<cost_functor_dynamic_numeric, ceres::CENTRAL>(
            new cost_functor_dynamic_numeric(dim, coefs) );
    cost_function->AddParameterBlock(dim);
    cost_function->SetNumResiduals(dim);
    return cost_function;
}

double run_test( const char *test_name, ceres::CostFunction *cost_function, int dim )
{
    double vec[dim] = { 0.0 };

    ceres::Problem problem;
    problem.AddResidualBlock( cost_function, NULL, vec );

    // Run the solver!
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    std::cout << test_name << std::endl;
    
    Solve( options, &problem, &summary );

    std::cout << summary.BriefReport() << "\n";

    double sum = 0.0;
    for( int i = 0; i < dim; i++ )
    {
        sum += (vec[i] - 1.0) * (vec[i] - 1.0);
    }
    sum /= dim;

    std::cout << "RMSE = " << sqrt(sum) << std::endl;

    return summary.total_time_in_seconds;
}

int main( int argc, char **argv )
{
    google::InitGoogleLogging( argv[0] );

    const int dim = DIM;

    /*
      Generate test data
     */
    int num_data = dim * (dim - 1);
    double *coefs = new double[num_data];

    for( int i = 0; i < num_data; i++ )
    {
        coefs[i] = ((double) rand()) / RAND_MAX - 0.5;
    }

    /*
      Run test
     */
    double time_analytic = run_test( "Test Analytic",
                                     new analytic_cost_function( coefs ),
                                     dim );
    double time_dynamic_analytic = run_test( "Test Dynamic Analytic",
                                             new dynamic_analytic_cost_function( dim, coefs ),
                                             dim );
    double time_autodiff = run_test( "Test Autodiff",
                                     new ceres::AutoDiffCostFunction<cost_functor_autodiff, DIM, DIM>(
                                         new cost_functor_autodiff( coefs ) ),
                                     dim );
    double time_dynamic_autodiff = run_test( "Test Dynamic Autodiff",
                                             gen_dynamic_autodiff_cost_function( dim, coefs ),
                                             dim );
    double time_numeric = run_test( "Test Numeric",
                                    new ceres::NumericDiffCostFunction<cost_functor_numeric,
                                                                       ceres::CENTRAL, DIM, DIM>(
                                        new cost_functor_numeric( coefs ) ),
                                    dim );
    double time_dynamic_numeric = run_test( "Test Dynamic Numeric",
                                            gen_dynamic_numeric_cost_function( dim, coefs ),
                                            dim );

    std::cout << "Total time (sec)\n";
    std::cout << "Analytic: " << time_analytic << std::endl;
    std::cout << "Dynamic Analytic: " << time_dynamic_analytic << std::endl;
    std::cout << "Autodiff: " << time_autodiff << std::endl;
    std::cout << "Dynamic Autodiff: " << time_dynamic_autodiff << std::endl;
    std::cout << "Numeric: "  << time_numeric << std::endl;
    std::cout << "Dynamic Numeric: " << time_dynamic_numeric << std::endl;
    
    delete[] coefs;

    return 0;
}
