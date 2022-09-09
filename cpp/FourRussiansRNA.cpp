//
//  FourRussiansRNA.cpp
//  FourRussiansRNA
//
//  Created by Yann Dubois on 01.04.17.
//  Copyright © 2017 Yann Dubois. All rights reserved.
//

#include "FourRussiansRNA.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <boost/python/numpy.hpp>
//CONSTANTS
const string folder("./Data/");

//CODE

// scoring function
int scoreB(char a, char b){
    // returns 1 if match 0 else
    if ((a == 'a' && b == 'u') || (a == 'u' && b == 'a') || (a == 'c' && b == 'g') || (b == 'c' && a == 'g')){
        //cout << " return " << 1 << endl;
        return 1;
    }
    return 0;
}
int C(int a, int b){
    if ((a & b) > 0){
        return 1;
    }
    return 0;
}
// preprocessing helper
int maxVal(ull x,ull y, const size_t q) {
    int max = 0, sum1 = 0, sum2 = 0;
    for (int k = int(q-1); k >= 0; k--) {
        if ((x & (1 << k)) != 0) sum1 = sum1 + 1;
        if ((y & (1 << k)) != 0) sum2 = sum2 - 1;
        if (sum1 + sum2 > max) max = sum1 + sum2;
    }
    return max;
}

void debugPrint(vector<vector<double> > &m) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            cout << m[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}


boost::python::numpy::ndarray convert_to_numpy(const vector<vector<double>> & input)
{
    u_int n_rows = input.size();
    u_int n_cols = input[0].size();
    boost::python::numpy::initialize();
    boost::python::tuple shape = boost::python::make_tuple(n_rows, n_cols);
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<double>();
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

//Four russians running
double nussinovFourRussians(const boost::python::list &seq, vector<vector<double>> &D, const size_t n, const int qParam, vector<pair<int, int>> &pairs  ){
    // INITIALIZATION
    size_t q;
    if (qParam == -1){
        // uses log base 2
        // note: log_a(x) = log(x)/log(a)
        q = round(log(n)/log(2));
    } else {
        q = qParam;
    }
    //Index
    //vector<vector<size_t>> Index (m,vector<size_t> (n));
    
    // preprocessing step for table R
    size_t qsq = pow(2, q);
    vvi R(qsq, vi(qsq));
    for (ull x = 0; x < (1 << q); ++x) {
        for (ull y = 0; y < (1 << q); ++y) {
            // x and y are horizontal and vertical difference bit vectors
            // represented by unsigned long longs
            R[x][y] = maxVal(x, y, q);
        }
    }
    
    int max_q = int(ceil(n/q))+1;
    vvu hvs(n, vu(max_q)); // horizontal diff vector store
    vvu vvs(n, vu(max_q)); // vertical diff vector store
    int reversed[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

    // ITERATION
    for (size_t j(0); j < n; ++j){
     pair<int, int> eventual = {-1, -1};
    int local_pairing = 0;
    int local_max = 0;
        for (long int i(j); i >= 0; --i){
            if (j - i <= 4) {
                D[i][j] = 0;
                continue;
            }
            local_max = max(D[i+1][j], D[i][j-1]);
            local_pairing = C(boost::python::extract<int>(seq[i]),reversed[boost::python::extract<int>(seq[j])]) + D[i+1][j-1];
            if (local_max >= local_pairing){
                D[i][j] = local_max;
                }
            else {
                    D[i][j] = local_pairing;
                    eventual = {i,j};
                }
            size_t groupI ((i)/q);
            size_t groupJ ((j)/q);
            // the matrix is diagonal so groupI doesn't start always from the same position but from
            // the diagonal
            int nGroupsBetween (int(groupJ - groupI - 1));
            // + q to get right most. -1 to put back in 0 index
            size_t iI = min(q*groupI + q - 1, n-2);
            size_t jJ(q*groupJ);
            // for all cells in the first group
            for (size_t k(i); k <= iI; ++k){
                if (D[i][k] + D[k+1][j] > D[i][j]){
                    D[i][j] =  D[i][k] + D[k+1][j];
                    eventual = {-1,-1};
                }
            }
            //for all cells in last group
            for (size_t k(jJ+1); k <= j; ++k){
                if(D[i][k-1] + D[k][j] > D[i][j]){
                    D[i][j] = D[i][k-1] + D[k][j];
                    eventual = {-1,-1};
                }
            }
            for (int K(0); K < nGroupsBetween; ++K){
                // take right most element of group I, adds 1 to get left most of next
                // then adds K*q to shift depedning on the block
                size_t l(iI + 1 + K*q);
                size_t t(l+1);
                ull hdiff = hvs[i][(iI + q*K + 1)/q];
                ull vdiff = vvs[j][t/q];
                if(D[i][l]+D[t][j]+R[hdiff][vdiff] > D[i][j]){
                    D[i][j] = D[i][l]+D[t][j]+R[hdiff][vdiff];
                    eventual = {-1,-1};
                }

            }
            
            // compute the vertical difference vector
            if (i % q == 1 && i + q <= n) {
                // compute and store the v¯ vector i/qth group for column j
                ull vdiff = 0; size_t c = q-2;
                for (size_t k(i); k < i+q-1; ++k) {
                    if (D[k][j] - D[k+1][j] == 1) {
                        vdiff = (vdiff | (1 << c));
                    } c--;
                }
                vvs[j][groupI] = vdiff; // i/qth group for column j
            }
            
            // compute the horizontal difference vector
            if (j % q == q - 1) {
                // compute and store the v vector (j − 1)/qth group for row i
                ull hdiff = 0; size_t c = q-2;
                for (size_t k(j+1-q); k <= j; ++k) {
                    if (D[i][k+1] - D[i][k] == 1) {
                        hdiff = (hdiff | (1 << c));
                    } c--;
                }
                hvs[i][groupJ] = hdiff; // (j − 1)/qth group for row i
            }

        }
    if (eventual.first != -1){
        pairs.push_back(eventual);
        }
    }

    return D[0][n-1];
    //TRACEBACK
}

string LoadSeq(string file){
    
    ifstream in(folder + file);
    
    if (!in) {
        cout << "Cannot open file.\n";
        return "1";
    }
    string str { istreambuf_iterator<char>(in), istreambuf_iterator<char>() };
    
    in.close();
    
    return str;
}

boost::python::list convert_list(const vector<pair<int, int>> & pairs){
    boost::python::list l;
    for(int i=0; i<pairs.size(); i++){
        l.append(boost::python::make_tuple(pairs[i].first, pairs[i].second));
    }
    return l;
}

boost::python::tuple FRScore(const boost::python::list & x, int n, int q) {
    vector<vector<double>> matrix(n, vector<double>(n, -1));
    vector<pair<int, int>> pairs;
    double score = nussinovFourRussians(x, matrix, n, q, pairs);
    
    boost::python::numpy::ndarray np_array = convert_to_numpy(matrix);
    boost::python::list pairings = convert_list(pairs);
    return boost::python::make_tuple(np_array, pairings);
}

BOOST_PYTHON_MODULE(four_russians){
    boost::python::def("four_russians", &FRScore);
}