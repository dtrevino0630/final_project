// Use this to compile: icpx -std=c++23 final_project.cpp

#include <iostream>
using std::cout;

#include <vector>
using std::vector;

#include <algorithm>
using std::max;

#include <span>
using std::span;

#include "mdspan/mdspan.hpp"
using Kokkos::mdspan;
using Kokkos::dextents;

#include <chrono>

// Exercise 53.10 METHODS End

// Exercise 53.2 Matrix Class
class Matrix {
  // Stores a Matrix as a vector of vectors
private:
  vector<vector<int>> data;
  int rows, cols;
public:
  // Constructor for empty matrix with given dimensions
  Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows, vector<int>(cols, 0))
  {}

  // Consturctor from a 2D vector
  Matrix(vector<vector<int>> data)
    : rows(data.size()), cols(data[0].size()), data(data)
  {}

  // Get number of rows
  int row_size() const { return rows; }

  // Get number of columns
  int col_size() const { return cols; }

  // Access particular element
  const int& at(int i, int j) const { return data[i][j]; }
  int& at(int i, int j) { return data[i][j]; }

  // Print out matrix
  void print() {
    for (auto& row : data) {
      for (auto& elem : row) {
        cout << elem << " ";
      }
      cout << '\n';
    }
  }

  // Traditional Matrix-Matrix Multiplication
  Matrix MatMult(const Matrix& other) {
    if (cols != other.rows)
      throw std::invalid_argument("Matrix dimensions do not allow multiplication.");

    // Creates matrix object to store result
    Matrix result(rows, other.cols);

    // Multiplies Matrices
    for (auto i = 0; i < rows; ++i) {
      for (auto j = 0; j < other.cols; ++j) {
        for (auto k = 0; k < cols; ++k) {
          result.at(i, j) += data[i][k] * other.at(k,j);
        }
      }
    }
    return result;
  }

  // Extract submatrix from (rowStart, colStart) of size (subRows x subCols)
  Matrix subMatrix(int rowStart, int colStart, int subRows, int subCols) const {
    Matrix result(subRows, subCols);
    for (auto i = 0; i < subRows; ++i) {
      for (auto j = 0; j < subCols; ++j) {
        result.at(i, j) = data[rowStart + i][colStart + j];
      }
    }
    return result;
  }

  // Submatrix back to the matrix at (rowStart, colStart)
  void writeSubMatrix(int rowStart, int colStart, const Matrix& subMat) {
    for (auto i = 0; i < subMat.row_size(); ++i) {
      for (auto j = 0; j < subMat.col_size(); ++j) {
        data[rowStart + i][colStart + j] = subMat.at(i, j);
      }
    }
  }

  // Add Matrices together
  Matrix addMatrices(const Matrix& A, const Matrix& B) const {
    if (A.row_size() != B.row_size() || A.col_size() != B.col_size()) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    Matrix result(A.row_size(), A.col_size());
    for (int i = 0; i < A.row_size(); ++i) {
      for (int j = 0; j < A.col_size(); ++j) {
        result.at(i, j) = A.at(i, j) + B.at(i, j);
      }
    }
    return result;
  }

  // Matrix Padding
  Matrix padMatrix(const Matrix& mat) const {
    int n = max(mat.row_size(), mat.col_size());
    int size = 1;

    while (size < n) size *= 2;
    Matrix padded(size, size);
    for (auto i = 0; i < mat.row_size(); ++i) {
      for (auto j = 0; j < mat.col_size(); ++j) {
        padded.at(i, j) = mat.at(i, j);
      }
    }
    return padded;
  }

  // Recursive Matrix-Matrix Multiplication
  Matrix MatMultRecur(const Matrix& other) const {
    if (cols != other.rows)
      throw std::invalid_argument("Matrix dimensions do not allow multiplication.");

    Matrix A = padMatrix(*this);
    Matrix B = padMatrix(other);
    int n = A.row_size();

    Matrix result(n, n);
    if (n == 1) {
      // Base case: 1x1 Matrix Multiplication
      result.at(0, 0) = data[0][0] * other.at(0, 0);
      return result;
    }

    int mid = n / 2;

    // Divide Matrices into 4 submatrices
    Matrix A11 = A.subMatrix(0, 0, mid, mid);
    Matrix A12 = A.subMatrix(0, mid, mid, n - mid);
    Matrix A21 = A.subMatrix(mid, 0, n - mid, mid);
    Matrix A22 = A.subMatrix(mid, mid, n - mid, n - mid);

    Matrix B11 = B.subMatrix(0, 0, mid, mid);
    Matrix B12 = B.subMatrix(0, mid, mid, n - mid);
    Matrix B21 = B.subMatrix(mid, 0, n - mid, mid);
    Matrix B22 = B.subMatrix(mid, mid, n - mid, n - mid);

    // Recursive multiplication and addition of submatrices
    Matrix C11 = addMatrices(A11.MatMultRecur(B11), A12.MatMultRecur(B21));
    Matrix C12 = addMatrices(A11.MatMultRecur(B12), A12.MatMultRecur(B22));
    Matrix C21 = addMatrices(A21.MatMultRecur(B11), A22.MatMultRecur(B21));
    Matrix C22 = addMatrices(A21.MatMultRecur(B12), A22.MatMultRecur(B22));

    // Combine results into one matrix
    result.writeSubMatrix(0, 0, C11);
    result.writeSubMatrix(0, mid, C12);
    result.writeSubMatrix(mid, 0, C21);
    result.writeSubMatrix(mid, mid, C22);

    // Extract the result submatrix of original dimensions
    return result.subMatrix(0, 0, rows, other.col_size());

  }
};

// Exercise 53.4 Matrix_Span Class
class Matrix_Span {
private:
  int m, lda, n; // Rows (m), Columns (n), Leading Dimension of A (lda)
  span<double> data; // Non-owning view of data
public:
  // Constructer for top-level matrix
  Matrix_Span(int m, int lda, int n, double* data)
    : m(m), lda(lda), n(n), data(data, lda * n)
  {}

  // Print matrix
  void print() const {
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        cout << this->at(i, j) << " ";
      }
      cout << '\n';
    }
  }

  // Exercise 53.4 METHOD Start

  double& at (int i, int j) const {
    // Exercise 53.3 ANSWER
    return data[j * lda + i];
  }

  // Exercise 53.4 METHOD End

  // Exercise 53.6 METHOD Start

  Matrix_Span addMatrices(const Matrix_Span& other, vector<double>& result_data, const Matrix_Span& result) const {
    if (m != other.m || n != other.n) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        result.at(i, j) = this->at(i, j) + other.at(i, j);
      }
    }
    return result;
  }

  // Exercise 53.6 METHOD End

  // Exercise 53.7 METHODS Start


  // Indexing with #define
#define AT(i, j) data[(j) * lda + (i)]

  // Matrix Addition with Debug and Optimized Access
  void def_addMat(const Matrix_Span& A, const Matrix_Span& B) {
    if (m != A.m ||  n != A.n || m != B.m || n != B.n) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        at(i, j) = A.at(i, j) + B.at(i, j);
      }
    }
  }

  // Print Matrix with def
  void def_print() const {
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        #ifdef DEBUG
        cout << at(i, j) << " ";
        #else
        cout << const_cast<double*>(data.data())[j * lda + i] << " ";
        #endif
      }
      cout << '\n';
    }
  }

  // Exercise 53.7 METHODS End

  // Exercise 53.8 METHODS Start

  //Block Submatrices
  // Top rows | Left cols
  Matrix_Span TopLeft(int rows, int cols) const {
    return Matrix_Span(rows, lda, cols, data.data());
  }

  // Top rows | Right cols
  Matrix_Span TopRight(int rows, int cols) const {
    return Matrix_Span(rows, lda, n - cols, data.data() + cols * lda);
  }

  // Bottom rows | Left cols
  Matrix_Span BotLeft(int rows, int cols) const {
    return Matrix_Span(m - rows, lda, cols, data.data() + rows);
  }

  // Bottom rows | Right cols
  Matrix_Span BotRight(int rows, int cols) const {
    return Matrix_Span(m - rows, lda, n - cols, data.data() + rows + cols * lda);
  }

  // Exercise 53.8 METHODS End

  // Exercise 53.9 METHODS Start

  // General Matrix Multiplication
  void MatMult(Matrix_Span& other, Matrix_Span& out) const {
    if (n != other.m || m != out.m || other.n != out.n) {
      throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
    }

    // Zero initialize output Matrix
    for (auto i = 0; i < out.m; ++i) {
      for (auto j = 0; j < out.n; ++j) {
        out.at(i, j) = 0.0;
      }
    }

    // Perform Matrix Multiplication
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < other.n; ++j) {
        for (auto k = 0; k < n; ++k) {
          out.at(i, j) += at(i, k) * other.at(k, j);
        }
      }
    }
  }

  // Block Matrix Multiplication (One Level Block)
  void BlockedMatMult(Matrix_Span& other, Matrix_Span& out) const {
    if (n != other.m || m != out.m || other.n != out.n) {
      throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
    }

    // Divide Matrices into Submatricies
    int midM = m / 2;
    int midN = n / 2;
    int midP = other.n / 2;

    Matrix_Span A11 = TopLeft(midM, midN);
    Matrix_Span A12 = TopRight(midM, midN);
    Matrix_Span A21 = BotLeft(midM, midN);
    Matrix_Span A22 = BotRight(midM, midN);

    Matrix_Span B11 = other.TopLeft(midN, midP);
    Matrix_Span B12 = other.TopRight(midN, midP);
    Matrix_Span B21 = other.BotLeft(midN, midP);
    Matrix_Span B22 = other.BotRight(midN, midP);

    Matrix_Span C11 = out.TopLeft(midM, midP);
    Matrix_Span C12 = out.TopRight(midM, midP);
    Matrix_Span C21 = out.BotLeft(midM, midP);
    Matrix_Span C22 = out.BotRight(midM, midP);

    // Temporary matrices to store intermediate results
    vector<double> temp1_data(midM * midP);
    vector<double> temp2_data(midM * midP);
    Matrix_Span temp1(midM, midP, midP, temp1_data.data());
    Matrix_Span temp2(midM, midP, midP, temp2_data.data());

    //assert(temp1_data.size() >= midM * midP);
    //assert(temp2_data.size() >= midM * midP);

    // Compute the 2 × 2 block matrix multiplication
    A11.MatMult(B11, temp1);
    A12.MatMult(B21, temp2);
    C11.def_addMat(temp1, temp2);

    A11.MatMult(B12, temp1);
    A12.MatMult(B22, temp2);
    C12.def_addMat(temp1, temp2);

    A21.MatMult(B11, temp1);
    A22.MatMult(B21, temp2);
    C21.def_addMat(temp1, temp2);

    A21.MatMult(B12, temp1);
    A22.MatMult(B22, temp2);
    C22.def_addMat(temp1, temp2);
  }

  // Recursive Block Matrix Multiplication
  void RecursiveMatMult(Matrix_Span& other, Matrix_Span& out) const {
    if (n != other.m || m != out.m || other.n != out.n) {
      throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
    }

    // Base case: Use regular matrix multiplication for small matrices
    if (m <= 2 || n <= 2 || other.n <= 2) {
      MatMult(other, out);
      return;
    }

    // Divide matrices into submatrices
    int midM = m / 2;
    int midN = n / 2;
    int midP = other.n / 2;

    Matrix_Span A11 = TopLeft(midM, midN);
    Matrix_Span A12 = TopRight(midM, midN);
    Matrix_Span A21 = BotLeft(midM, midN);
    Matrix_Span A22 = BotRight(midM, midN);

    Matrix_Span B11 = other.TopLeft(midN, midP);
    Matrix_Span B12 = other.TopRight(midN, midP);
    Matrix_Span B21 = other.BotLeft(midN, midP);
    Matrix_Span B22 = other.BotRight(midN, midP);

    Matrix_Span C11 = out.TopLeft(midM, midP);
    Matrix_Span C12 = out.TopRight(midM, midP);
    Matrix_Span C21 = out.BotLeft(midM, midP);
    Matrix_Span C22 = out.BotRight(midM, midP);

    // Temporary matrices to store intermediate results
    vector<double> rec1_data(midM * midP);
    vector<double> rec2_data(midM * midP);
    Matrix_Span rec1(midM, midP, midP, rec1_data.data());
    Matrix_Span rec2(midM, midP, midP, rec2_data.data());

    // Recursive computations for 2 × 2 block product
    A11.RecursiveMatMult(B11, rec1);
    A12.RecursiveMatMult(B21, rec2);
    C11.def_addMat(rec1, rec2);

    A11.RecursiveMatMult(B12, rec1);
    A12.RecursiveMatMult(B22, rec2);
    C12.def_addMat(rec1, rec2);

    A21.RecursiveMatMult(B11, rec1);
    A22.RecursiveMatMult(B21, rec2);
    C21.def_addMat(rec1, rec2);

    A21.RecursiveMatMult(B12, rec1);
    A22.RecursiveMatMult(B22, rec2);
    C22.def_addMat(rec1, rec2);
  }

  // Exercise 53.9 METHODS End
};

  // Exercise 53.5 Matrix_Mdspan Class

  class Matrix_Mdspan {
private:
  int m, lda, n; // Rows (m), Columns (n), Leading Dimension of A (lda)
    mdspan<double, dextents<int, 2>> data; // Non-owning view of data
public:
  // Constructer for top-level matrix
  Matrix_Mdspan(int m, int lda, int n, double* ptr)
    : m(m), lda(lda), n(n), data(ptr, lda, n)
  {}

    // Print matrix
    void print() const {
      for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < n; ++j) {
          cout << this->at(i, j) << " ";
        }
        cout << '\n';
      }
    }

    // Access Matrix
    double& at (int i, int j) const {
      return data[i, j];
    }
  };

int main() {

  // Define Timer
  using timer = std::chrono::steady_clock;
  timer::time_point before = timer::now();
  auto after = timer::now();

  // Exercise 53.1 CODE Start

  // Define matrices using vectors
  vector<vector<int>> a = {{1, 2, 3}, {4, 5, 6}};
  vector<vector<int>> b = {{1, 2}, {3, 4}, {5, 6}};
  vector<vector<int>> c(a.size(), vector<int>(b[0].size(), 0)); // Initialize result matrix with zeros
  vector<vector<int>> d(a.size(), vector<int>(b[0].size(), 0)); // Initialize result matrix with zeros

  // Perform matrix multiplication
  before = timer::now();
  for (int i = 0; i < a.size(); i++) { // Rows of a
      for (int j = 0; j < b[0].size(); j++) { // Columns of b
          for (int k = 0; k < a[0].size(); k++) { // Columns of a / Rows of b
              c[i][j] += a[i][k] * b[k][j];
          }
      }
  }
  after = timer::now();

  // Output result
  cout << "Straightforward Output" << '\n';
  for (int i = 0; i < c.size(); i++) {
      for (int j = 0; j < c[0].size(); j++) {
          cout << c[i][j] << " ";
      }
      cout << "\n";
  }
  cout << "Straightforward Output took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  // Permuted Algorithm to Perform matrix multiplication
  before = timer::now();
  for (int j = 0; j < b[0].size(); j++) { // Columns of b
        for (int i = 0; i < a.size(); i++) { // Rows of a
            for (int k = 0; k < a[0].size(); k++) { // Columns of a / Rows of b
                d[i][j] += a[i][k] * b[k][j];
            }
        }
  }
  after = timer::now();

  // Permuted Algorithm Output result
  cout << "\nPermuted Algorithm Output:" << '\n';
  for (int i = 0; i < d.size(); i++) {
      for (int j = 0; j < d[0].size(); j++) {
          cout << d[i][j] << " ";
      }
      cout << "\n";
  }
  cout << "Permuted Algorithm took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  // Exercise 53.1 CODE End

  // Exercise 53.2 CODE Start

  Matrix A({{1, 2, 3}, {4, 5, 6}});
  Matrix B({{1, 2}, {3, 4}, {5, 6}});

  cout << "\nMatrix A:\n";
  A.print();

  cout << "\nMatrix B:\n";
  B.print();

  cout << "\nTraditional Matrix-Matrix Multiplication (A * B):\n";
  before = timer::now();
  Matrix C = A.MatMult(B);
  after = timer::now();
  C.print();
  cout << "Traditional Matrix-Matrix Multiplication took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  cout << "\nRecursive Matrix-Matrix Multiplication (A * B):\n";
  Matrix D = A.MatMultRecur(B);
  before = timer::now();
  D.print();
  after = timer::now();
  cout << "Recursive Matrix-Matrix Multiplication took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";
  // Exercise 53.2 CODE End

  // Exercise 53.4 CODE Start

  // Example values for m, n, lda
  int m = 2; // Number of rows
  int n = 3; // Number of columns
  int lda = m + 2; // Leading Dimension

  // Create vector to contain values
  vector<double> data_1(lda * n, 1.0);

  // Create matrix using vector data
  Matrix_Span one(m, lda, n, data_1.data());

  // Print top-level matrix
  cout << "\nTop-Level Matrix:\n";
  one.print();

  // Modify element within matrix
  one.at(1, 0) = 99;

  // Print updated top-level matrix
  cout << "\nUpdated Top-Level Matrix\n";
  one.print();

  // Print out value at i = 0 and j = 1 for top-level matrix
  cout << "\nTop-Level Matrix (0, 1): " << one.at(0, 1);

  // Exercise 54.3 CODE End

  // Exercise 54.6 CODE Start

  // Example values for m, n, lda
  int m_2 = 3; // Number of rows
  int n_2 = 3; // Number of columns
  int lda_2 = m + 2; // Leading Dimension
  int m_3 = 3;
  int n_3 = 3;
  int lda_3 = m + 1;

  // Create vector to contain values
  vector<double> data_2(lda_2 * n_2, 1.0);
  vector<double> data_3(lda_3 * n_3, 2.0);
  vector<double> result_data(lda_2 * n_2, 0.0);

  // Create matrix using vector data
  Matrix_Span two(m_2, lda_2, n_2, data_2.data());
  Matrix_Span three(m_3, lda_3, n_3, data_3.data());
  Matrix_Span result(m_2, lda_2, n_2, result_data.data());

  // Print each matrix
  cout << "\nMatrix Two:\n";
  two.print();

  cout << "\nMatrix Three:\n";
  three.print();

  // Add matrices together
  before = timer::now();
  Matrix_Span four = two.addMatrices(three, result_data, result);
  after = timer::now();

  // Print added matrix
  cout <<"\nAdded Matrix (Two + Three):\n";
  four.print();
  cout << "Adding Two Matrices took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  // Exercise 53.6 CODE End

  // Exercise 53.7 CODE Start

  int m_def = 3;
  int n_def = 3;
  int lda_def = 4;

  // Create storage for matrices
  vector<double> dataA(lda_def * n_def, 1.0);
  vector<double> dataB(lda_def * n_def, 2.0);
  vector<double> dataC(lda_def * n_def, 0.0);

  // Create Top-Level Matrices
  Matrix_Span A_def(m_def, lda_def, n_def, dataA.data());
  Matrix_Span B_def(m_def, lda_def, n_def, dataB.data());
  Matrix_Span C_def(m_def, lda_def, n_def, dataC.data());

  // Print Matrices
  cout << "\nMatrix A_def:\n";
  A_def.def_print();

  cout << "\nMatrix B_def:\n";
  B_def.def_print();

  cout << "\nMatrix C_def:\n";
  C_def.def_print();

  // Add Matrices
  before = timer::now();
  C_def.def_addMat(A_def, B_def);
  after = timer::now();

  // Print Results
  cout << "\nAdded Matrix using Def (C_def = A_def + B_def):\n";
  C_def.def_print();
  cout << "Adding Two Matrices using Def took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  // Exercise 53.7 CODE End

  // Exercise 53.8 CODE Start

  int m_sub = 4;
  int n_sub = 6;
  int lda_sub = 6;

  // Create storage for matrix
  vector<double> data_sub(lda_sub * n_sub, 0.0);
  for (auto i = 0; i < m_sub; ++i) {
    for (auto j = 0; j < n_sub; ++j) {
      data_sub[j * lda_sub + i] = i + j * 0.1; // Initialize with unique values
    }
  }

  // Create Top-Level Matrix
  Matrix_Span S(m_sub, lda_sub, n_sub, data_sub.data());

  // Print Matrix
  cout << "\nOriginal Matrix S:\n";
  S.def_print();

  // Test Submatrix Methods
  cout << "\nTopLeft(2, 3):\n";
  Matrix_Span topLeft = S.TopLeft(2, 3);
  topLeft.print();

  cout << "\nTopRight(2, 3):\n";
  Matrix_Span topRight = S.TopRight(2, 3);
  topRight.print();

  cout << "\nBotLeft(2, 3):\n";
  Matrix_Span botLeft = S.BotLeft(2, 3);
  botLeft.print();

  cout << "\nBotRight(2, 3):\n";
  Matrix_Span botRight = S.BotRight(2, 3);
  botRight.print();

  // Exercise 53.8 CODE End

  // Exercise 53.9 CODE Start

  int m_mult = 4;
  int n_mult = 4;
  int lda_mult = 4;

  // Create storage for matrix
  vector<double> data_mult1(lda_mult * n_mult, 0.0);
  vector<double> data_mult2(lda_mult * n_mult, 0.0);
  vector<double> temp_mult1(lda_mult * n_mult, 0.0);
  vector<double> temp_mult2(lda_mult * n_mult, 0.0);
  vector<double> temp_mult3(lda_mult * n_mult, 0.0);
  for (auto i = 0; i < m_mult; ++i) {
    for (auto j = 0; j < n_mult; ++j) {
      data_mult1[j * lda_mult + i] = i + j * 1; // Initialize with unique values
      data_mult2[j * lda_mult + i] = i + j * 2; // Initialize with unique values
    }
  }

  // Create Top-Level Matrices
  Matrix_Span A_mult(m_mult, lda_mult, n_mult, data_mult1.data());
  Matrix_Span B_mult(m_mult, lda_mult, n_mult, data_mult2.data());
  Matrix_Span D_mult(m_mult, lda_mult, n_mult, temp_mult1.data());
  Matrix_Span E_mult(m_mult, lda_mult, n_mult, temp_mult2.data());
  Matrix_Span F_mult(m_mult, lda_mult, n_mult, temp_mult3.data());

  // Print Matrices
  cout << "\nMatrix A_mult:\n";
  A_mult.def_print();

  cout << "\nMatrix B_mult:\n";
  B_mult.def_print();

  cout << "\nMatrix D_mult:\n";
  D_mult.def_print();

  cout << "\nMatrix E_mult:\n";
  E_mult.def_print();

  cout << "\nMatrix F_mult:\n";
  F_mult.def_print();

  before = timer::now();
  // Perform Traditional Matrix Multiplication
  A_mult.MatMult(B_mult, D_mult);
  after = timer::now();
  // Print Results
  cout << "\nResulting Matrix D_mult (A_mult * B_mult):\n";
  D_mult.def_print();
  cout << "Traditional Method took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";
  before = timer::now();

  // Perform Blocked Matrix Multiplication
  A_mult.BlockedMatMult(B_mult, E_mult);
  after = timer::now();
  // Print Results
  cout << "\nResulting Matrix E_mult (A_mult * B_mult):\n";
  E_mult.def_print();
  cout << "Block Method took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  before = timer::now();
  A_mult.RecursiveMatMult(B_mult, F_mult);
  after = timer::now();
  // Print Results
  cout << "\nResulting Matrix F_mult (A_mult * B_mult):\n";
  F_mult.def_print();
  cout << "Recursive Method took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  // Testing Speed of 8x8 Matrix
  int m_t = 8;
  int n_t = 8;
  int lda_t = 8;

  // Create storage for matrix
  vector<double> data_t1(lda_t * n_t, 0.0);
  vector<double> data_t2(lda_t * n_t, 0.0);
  vector<double> temp_t1(lda_t * n_t, 0.0);
  vector<double> temp_t2(lda_t * n_t, 0.0);
  vector<double> temp_t3(lda_t * n_t, 0.0);
  for (auto i = 0; i < m_t; ++i) {
    for (auto j = 0; j < n_t; ++j) {
      data_t1[j * lda_t + i] = i + j; // Initialize with unique values
      data_t2[j * lda_t + i] = i - j; // Initialize with unique values
    }
  }

  // Create Top-Level Matrices
  Matrix_Span A_t(m_t, lda_t, n_t, data_t1.data());
  Matrix_Span B_t(m_t, lda_t, n_t, data_t2.data());
  Matrix_Span D_t(m_t, lda_t, n_t, temp_t1.data());
  Matrix_Span E_t(m_t, lda_t, n_t, temp_t2.data());
  Matrix_Span F_t(m_t, lda_t, n_t, temp_t3.data());

  before = timer::now();
  // Perform Traditional Matrix Multiplication
  A_t.MatMult(B_t, D_t);
  after = timer::now();

  // Print Resulting Matrix
  cout << "\nResulting Matrix D_mult (A_mult * B_mult):\n";
  D_t.def_print();
  cout << "Traditional Method took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  before = timer::now();
  // Perform Blocked Matrix Multiplication
  A_t.BlockedMatMult(B_t, E_t);
  after = timer::now();
  // Print Resulting Matrix
  cout << "\nResulting Matrix E_mult (A_mult * B_mult):\n";
  E_t.def_print();
  cout << "Block Method took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  before = timer::now();
  A_t.RecursiveMatMult(B_t, F_t);
  after = timer::now();
  // Print Resulting Matrix
  cout << "\nResulting Matrix F_mult (A_mult * B_mult):\n";
  F_t.def_print();
  cout << "Recursive Method took: " << duration_cast<std::chrono::nanoseconds>(after-before).count() << " ns\n";

  // Print Matrices
  cout << "\nMatrix A_mult:\n";
  A_t.def_print();

  cout << "\nMatrix B_mult:\n";
  B_t.def_print();

  return 0;
}
