#include <vector>
#include <boost/python/numpy.hpp>
#include <iostream>

using namespace std;

int reverseBits(int num)
{
    int count = 4;
    int reverse_num = 0;
       
    while(count)
    {
       reverse_num <<= 1;       
       reverse_num |= num & 1;
       num >>= 1;
       count--;
    }
    return reverse_num;
}

boost::python::numpy::ndarray convert_to_numpy(const vector<vector<int>> & input)
{
    u_int n_rows = input.size();
    u_int n_cols = input[0].size();
    boost::python::numpy::initialize();
    boost::python::tuple shape = boost::python::make_tuple(n_rows, n_cols);
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();
    boost::python::numpy::ndarray ndar = boost::python::numpy::zeros(shape, dt);

    for (u_int i = 0; i < n_rows; i++)
    {
        for (u_int j = 0; j < n_cols; j++)
        {
            ndar[i][j] = input[i][j];
        }
    }
    
    return ndar;
}

int C(int a, int b){
    if ((a & b) > 0){
        return 1;
    }
    return 0;
}

int foldScoreIterative(const boost::python::list &seq, vector<vector<int>> &matrix, int n) {
    int reversed[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    for (int j = 0; j < n; j++){
         for (int i = j; i >= 0; i--) {
             if (j - i <= 3) {
                 continue;
             }
             matrix[i][j] = max(max(
                                    matrix[i+1][j],
                                    matrix[i][j-1]),
                                matrix[i+1][j-1]+C(boost::python::extract<int>(seq[i]), reversed[boost::python::extract<int>(seq[j])]));
             for (int k = i + 1; k < j; k++) {
                 matrix[i][j] = max(
                                    matrix[i][j],
                                    matrix[i][k]+matrix[k+1][j]);
             }
         }
    }
    // Alternative recursion schemes
  
    // for (int d=4; d<n; d++){
    //     for(int i = 0; i < n-d; i++){
    //         int j = i+d;
    //         int temp = 0;
    //         for(int k = i; k<j; k++){
    //             temp = max(temp, matrix[i][k-1]+matrix[k+1][j-1] + (j-k>3 ? B(boost::python::extract<int>(seq[i]), boost::python::extract<int>(seq[j])) : 0));
    //         }
    //         matrix[i][j] = max(matrix[i][j-1], temp);
    //     }
    // }
    // for (int j=1; j<n;j++){
    //     for (int i=0; i<j-1; i++){
    //         matrix[i][j] = max(matrix[i+1][j-1]+ (j-i>3 ? B(boost::python::extract<int>(seq[i]), reversed[boost::python::extract<int>(seq[j])]) : 0), matrix[i][j-1]);
    //     }
    //     for (int i=j-2; i >= 0; i--){
    //         matrix[i][j] = max(matrix[i+1][j], matrix[i][j]);
    //         for (int k = j-2; k >= i; k--){
    //             matrix[i][j] = max(matrix[i][j], matrix[i][k-1] + matrix[k][j]);
    //         }
    //     }
    // }
    return matrix[0][n-1];
    
}

boost::python::numpy::ndarray nussinovScore(const boost::python::list & x, int n) {
    vector<vector<int>> matrix(n, vector<int>(n, 0));
    int score = foldScoreIterative(x, matrix, n);
    
    boost::python::numpy::ndarray np_array = convert_to_numpy(matrix);
    return np_array;
}

BOOST_PYTHON_MODULE(nussinov){
    boost::python::def("nussinov", &nussinovScore);
}
