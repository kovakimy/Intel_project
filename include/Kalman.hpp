#pragma once

#include <vector>
#include <opencv2/core/types.hpp>

void GetCofactor(std::vector<std::vector<float>> &A, std::vector<std::vector<float>> &temp, int p, int q, int n)
{
    int ig = 0, jg = 0;
    int n_tmp = temp.size();

    // Looping for each element of the matrix
    for (int row = 0; row < n_tmp; row++)
    {
        for (int col = 0; col < n_tmp; col++)
        {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q)
            {
                temp[ig][jg++] = A[row][col];

                // Row is filled, so increase row index and
                // reset col index
                if (jg == n_tmp - 1)
                {
                    jg = 0;
                    ig++;
                }
            }
        }
    }
}

/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][]. */
float Determinant(std::vector<std::vector<float>> &A, int n)
{
    size_t N_det = A.size();
    float D = 0; // Initialize result

    //  Base case : if matrix contains single element
    if (n == 1)
        return A[0][0];
    
    std::vector<std::vector<float>> temp(N_det, std::vector<float>(N_det, 0)); // To store cofactors

    float sign = 1;  // To store sign multiplier

     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        GetCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * Determinant(temp, n - 1);

        // terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}

// Function to get adjoint of A[N][N] in adj[N][N].
void Adjoint(std::vector<std::vector<float>> &A, std::vector<std::vector<float>> &adj)
{
    int N_adj = A.size();
    if (N_adj == 1)
    {
        adj[0][0] = 1;
        return;
    }

    // temp is used to store cofactors of A[][]
    int sign = 1;
    std::vector<std::vector<float>> temp(N_adj, std::vector<float>(N_adj, 0));

    for (int i = 0; i < N_adj; i++)
    {
        for (int j = 0; j < N_adj; j++)
        {
            // Get cofactor of A[i][j]
            GetCofactor(A, temp, i, j, N_adj);

            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j][i] = (sign) * (Determinant(temp, N_adj - 1));
        }
    }
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool mat_inverse(std::vector<std::vector<float>> &A, std::vector<std::vector<float>> &inverse)
{
    int N = A.size();
    // Find determinant of A[][]
    float det = Determinant(A, N);
    if (det == 0)
    {
        //cout << "Singular matrix, can't find its inverse";
        return false;
    }

    // Find adjoint
    std::vector<std::vector<float>> adj(N, std::vector<float>(N, 0));// [N] [N] ;
    Adjoint(A, adj);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inverse[i][j] = adj[i][j] / float(det);

    return true;
}


std::vector<std::vector<float>> matrix_mul(const std::vector<std::vector<float>> &u, const std::vector<std::vector<float>> &v)
{
	size_t n = u.size();
	size_t m = v[0].size();
	std::vector<std::vector<float>> result(n, std::vector<float>(m, 0));
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < m; ++j)
		{
			for (size_t k = 0; k < v.size(); ++k)
			{
				result[i][j] += (u[i][k] * v[k][j]);
			}
		}
	}
	return result;
}


std::vector<std::vector<float>> matrix_diff(const std::vector<std::vector<float>> &u, const std::vector<std::vector<float>> &v)
{
	size_t n = u.size();
	size_t m = u[0].size();
	std::vector<std::vector<float>> result(n, std::vector<float>(m, 0));
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < m; ++j)
		{
			result[i][j] = u[i][j] - v[i][j];
		}
	}
	return result;
}


std::vector<std::vector<float>> matrix_add(const std::vector<std::vector<float>> &u, const std::vector<std::vector<float>> &v)
{
	size_t n = u.size();
	size_t m = u[0].size();
	std::vector<std::vector<float>> result(n, std::vector<float>(m, 0));
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < m; ++j)
		{
			result[i][j] = u[i][j] + v[i][j];
		}
	}
	return result;
}


std::vector<std::vector<float>> m_transpose(const std::vector<std::vector<float> >& vec)
{
	size_t n = vec.size();
	size_t m = vec[0].size();
	std::vector<std::vector<float>> trans_vec(m, std::vector<float>(n, 0));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			trans_vec[j][i] = vec[i][j];
		}
	}
	return trans_vec;
}

cv::Point kalman(std::vector<std::vector<float>> &x, std::vector<std::vector<float>> &P, float obs_x, float obs_y, float R)
{
	std::vector<std::vector<float>> motion(4, std::vector<float>(1, 0));
	std::vector<std::vector<float>> Q(4, std::vector<float>(4, 0));
	Q[0][0] = 1;
	Q[1][1] = 1;
	Q[2][2] = 1;
	Q[3][3] = 1;
	std::vector<std::vector<float>> F(4, std::vector<float>(4, 0));
	F[0][0] = 1;
	F[1][1] = 1;
	F[2][2] = 1;
	F[3][3] = 1;
	F[0][2] = 1;
	F[1][3] = 1;
	std::vector<std::vector<float>> H(2, std::vector<float>(4, 0));
	H[0][0] = 1;
	H[1][1] = 1;
	
	
	std::vector<std::vector<float>> measurement(1, std::vector<float>(2, 0));
	measurement[0][0] = obs_x;
	measurement[0][1] = obs_y;

	// y = np.matrix(measurement).T - H * x
	std::vector<std::vector<float>> y = matrix_diff(m_transpose(measurement), matrix_mul(H, x));
	
	//S = H * P * H.T + R  # residual convariance
	std::vector<std::vector<float>> S = matrix_mul(matrix_mul(H, P), m_transpose(H));
	//S = matrix_add(S, R);
	
	for (size_t i = 0; i < S.size(); ++i)
	{
		for (size_t j = 0; j < S[0].size(); ++j)
		{
			S[i][j] += R;
		}
	}

	//K = P * H.T * S.I    # Kalman gain
	std::vector<std::vector<float>> S_inv(S.size(), std::vector<float>(S.size(), 0));
	//if (mat_inverse(S, S_inv))
	//	cout << "inversion OK" << endl;
	mat_inverse(S, S_inv);
	std::vector<std::vector<float>> K = matrix_mul(matrix_mul(P, m_transpose(H)), S_inv);

	//x = x + K*y 
	x = matrix_add(x, matrix_mul(K, y));

	//I = np.matrix(np.eye(F.shape[0])) # identity matrix
	std::vector<std::vector<float>> I(F.size(), std::vector<float>(F.size(), 0));
	I[0][0] = 1;
	I[1][1] = 1;
	I[2][2] = 1;
	I[3][3] = 1;

	//P = (I - K*H)*P
	P = matrix_mul(matrix_diff(I, matrix_mul(K, H)), P);

	//PREDICT x, P based on motion
	//x = F*x + motion
	x = matrix_add(matrix_mul(F, x), motion);

	//P = F*P*F.T + Q
	P = matrix_add(matrix_mul(matrix_mul(F, P), m_transpose(F)), Q);
	cv::Point res(x[0][0], x[1][0]);
	return res;
}