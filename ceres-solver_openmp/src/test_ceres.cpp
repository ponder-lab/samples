#include <omp.h>
#include <ceres/ceres.h>

/*
  Finding intersection of (dim) number of hyper planes

  a(x0 - 1) + b(x1 - 1) + ... + c(xn_2 - 1) + (xn_1 - 1) = 0
  d(x0 - 1) + e(x1 - 1) + ... + f(xn_2 - 1) + (xn_1 - 1) = 0
  ...
  g(x0 - 1) + h(x1 - 1) + ... + i(xn_2 - 1) + (xn_1 - 1) = 0

  should return x0 = x1 = ... = 1
 */
struct CostFunctor
{
    CostFunctor( int dim_ ) : dim( dim_ )
    {
        int num_data = dim_ * (dim_ - 1);
        coefs = new double[num_data];

        for( int i = 0; i < num_data; i++ )
        {
            coefs[i] = ((double) rand()) / RAND_MAX - 0.5;
        }
    }

    ~CostFunctor()
    {
        delete[] coefs;
    }

    template <typename T>
    bool operator()(T const* const* x, T* res) const
    {
#pragma omp parallel for
        for( int i = 0; i < dim; i++ )
        {
            res[i] = T(0.0);
            int j;
            for( j = 0; j < (dim - 1); j++ )
            {
                double coef = coefs[i*(dim-1)+j];
                res[i] += T(coef) * ( x[0][j] - T(1.0) );
            }
            res[i] += x[0][j] - T(1.0);
        }

        return true;
    }

    int dim;
    double *coefs;
};

int main( int argc, char **argv )
{
    google::InitGoogleLogging( argv[0] );

    const int dim = 1000;
    double vec[dim] = { 0.0 };

    ceres::Problem problem;

    ceres::DynamicAutoDiffCostFunction<CostFunctor>* cost_function =
        new ceres::DynamicAutoDiffCostFunction<CostFunctor>(
            new CostFunctor(dim) );
    cost_function->AddParameterBlock(dim);
    cost_function->SetNumResiduals(dim);
    problem.AddResidualBlock( cost_function, NULL, vec );

    // Run the solver!
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve( options, &problem, &summary );

    std::cout << summary.BriefReport() << "\n";

    double sum = 0.0;
    for( int i = 0; i < dim; i++ )
    {
        sum += (vec[i] - 1.0) * (vec[i] - 1.0);
    }
    sum /= dim;

    std::cout << "RMSE = " << sqrt(sum) << std::endl;
    
    return 0;
}
