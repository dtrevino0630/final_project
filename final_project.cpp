// Use this to compile: icpx -std=c++23 final_project.cpp

#include <iostream>
using std::cout;

#include <vector>
using std::vector;

#include <algorithm>
using std::max;

#include <span>
using std::span;
/**
#include "mdspan/mdspan.hpp"
using Kokkos::mdspan;
using Kokkos::dextents;
*/
#include <chrono>

// Exercise 53.10 METHODS Start

  template<typename Func, typename... Args>
  void timeExecution(const std::string& label, Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << label << " took " << elapsed.count() << " seconds.\n";
  }

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
  int& operator()(int i, int j) { return data[i][j]; }
  const int& operator()(int i, int j) const { return data[i][j]; }

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
          result(i,j) += data[i][k] * other(k,j);
        }
      }
    }
    return result;
  }

  // Recursive Matrix-Matrix Multiplication
  Matrix MatMultRecur(const Matrix& other) const {
    if (cols != other.rows)
      throw std::invalid_argument("Matrix dimensions do not allow multiplication.");

    // Determine the required size (next power of 2)
    int maxSize = max({row_size(), col_size(), other.row_size(), other.col_size()});
    int newSize = 1;
    while (newSize < maxSize) newSize *= 2;

    // Pad matrices to be square and power-of-2 sized
    Matrix paddedA = padMatrix(*this, newSize);
    Matrix paddedB = padMatrix(other, newSize);

    // Perform recursive multiplication on padded matrices
    Matrix paddedResult = recursiveHelper(paddedA, paddedB);

    // Extract the result submatrix of original dimensions
    return paddedResult.subMatrix(0, 0, row_size(), other.col_size());

  }

private:
  // Extract submatrix from (rowStart, colStart) of size (subRows x subCols)
  Matrix subMatrix(int rowStart, int colStart, int subRows, int subCols) const {
    Matrix result(subRows, subCols);
    for (auto i = 0; i < subRows; ++i) {
      for (auto j = 0; j < subCols; ++j) {
        result(i, j) = data[rowStart + i][colStart + j];
      }
    }
    return result;
  }

  // Submatrix back to the matrix at (rowStart, colStart)
  void writeSubMatrix(int rowStart, int colStart, const Matrix& subMat) {
    for (auto i = 0; i < subMat.row_size(); ++i) {
      for (auto j = 0; j < subMat.col_size(); ++j) {
        data[rowStart + i][colStart + j] = subMat(i, j);
      }
    }
  }

  Matrix addMatrices(const Matrix& A, const Matrix& B) const {
    if (A.row_size() != B.row_size() || A.col_size() != B.col_size()) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    Matrix result(A.row_size(), A.col_size());
    for (int i = 0; i < A.row_size(); ++i) {
      for (int j = 0; j < A.col_size(); ++j) {
        result(i, j) = A(i, j) + B(i, j);
      }
    }
    return result;
  }

  Matrix padMatrix(const Matrix& original, int newSize) const {
    Matrix padded(newSize, newSize);
    for (int i = 0; i < original.row_size(); ++i) {
      for (int j = 0; j < original.col_size(); ++j) {
        padded(i, j) = original(i, j);
      }
    }
    return padded;
  }

  // Helper function for recursive multiplication
  Matrix recursiveHelper(const Matrix& A, const Matrix& B) const {
    int n = A.row_size();
    if (n == 1) {
      // Base case: 1x1 Matrix Multiplication
      return Matrix({{A(0,0) * B(0, 0)}});
    }

    int mid = n / 2;

    // Divide Matrices into 4 submatrices
    auto A11 = A.subMatrix(0, 0, mid, mid);
    auto A12 = A.subMatrix(0, mid, mid, n - mid);
    auto A21 = A.subMatrix(mid, 0, n - mid, mid);
    auto A22 = A.subMatrix(mid, mid, n - mid, n - mid);

    auto B11 = B.subMatrix(0, 0, mid, mid);
    auto B12 = B.subMatrix(0, mid, mid, n - mid);
    auto B21 = B.subMatrix(mid, 0, n - mid, mid);
    auto B22 = B.subMatrix(mid, mid, n - mid, n - mid);

    // Recursive multiplication and addition of submatrices
    auto C11 = addMatrices(A11.MatMultRecur(B11), A12.MatMultRecur(B21));
    auto C12 = addMatrices(A11.MatMultRecur(B12), A12.MatMultRecur(B22));
    auto C21 = addMatrices(A21.MatMultRecur(B11), A22.MatMultRecur(B21));
    auto C22 = addMatrices(A21.MatMultRecur(B12), A22.MatMultRecur(B22));

    // Combine results into one matrix
    Matrix result(n, n);
    result.writeSubMatrix(0, 0, C11);
    result.writeSubMatrix(0, mid, C12);
    result.writeSubMatrix(mid, 0, C21);
    result.writeSubMatrix(mid, mid, C22);

    return result;
  }
};

// Exercise 53.4 Matrix_Span Class
class Matrix_Span {
private:
  int m, lda, n; // Rows (m), Columns (n), Leading Dimension of A (lda)
  span<double> data; // Non-owning view of data
public:
  // Constructer for top-level matrix
  Matrix_Span(int m, int lda, int n, double* ptr)
    : m(m), lda(lda), n(n), data(ptr, lda * n)
  {}

  // Constructor for submatrix
  Matrix_Span(int m, int lda, int n, span<double> parent, int rowOffset, int colOffset)
    : m(m), lda(lda), n(n), data(parent.subspan(colOffset * lda + rowOffset, m * n))
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

  double* get_double_data() const {
    return const_cast<double*>(data.data());
  }

  // Matrix Addition with Debug and Optimized Access
  void def_addMatrices(const Matrix_Span& A, const Matrix_Span& B) {
    if (m != A.m ||  n != A.n || m != B.m || n != B.n) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    double* cdata = get_double_data();
    double* adata = A.get_double_data();
    double* bdata = B.get_double_data();

// Define Indexing Macros
#define C_INDEX(i, j) cdata[(j) * lda + (i)]
#define A_INDEX(i, j) adata[(j) * A.lda + (i)]
#define B_INDEX(i, j) bdata[(j) * B.lda + (i)]

    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        #ifdef DEBUG
        at(i, j) = A.at(i, j) + B.at(i, j);
        #else
        C_INDEX(i, j) = A_INDEX(i, j) + B_INDEX(i, j);
        #endif
      }
    }

    // Undefine Macros to Avoid Conflict
    #undef C_INDEX
    #undef A_INDEX
    #undef B_INDEX
  }

  // Print Matrix with def
  void def_print() const {
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        #ifdef DEBUG
        cout << at(i, j) << " ";
        #else
        cout << get_double_data()[j * lda + i] << " ";
        #endif
      }
      cout << '\n';
    }
  }

  // Exercise 53.7 METHODS End

  // Exercise 53.8 METHODS Start

  // Submatrix methods
  Matrix_Span Left(int j) const { return Matrix_Span(m, lda, j, data, 0, 0); } // n = j (cols < j)
  Matrix_Span Right(int j) const { return Matrix_Span(m, lda, n - j, data, 0, j); } // n = n - j (cols >= j)
  Matrix_Span Top(int i) const { return Matrix_Span(i, lda, n, data, 0, 0); } // m = i (rows < i)
  Matrix_Span Bot(int i) const { return Matrix_Span(m - i, lda, n, data, i, 0); } // m = m - i (rows >= i)

  //Block Submatrices
  // Top rows | Left cols
  Matrix_Span TopLeft(int rows, int cols) const { return Matrix_Span(rows, lda, cols, data, 0, 0); }
  // Top rows | Right cols
  Matrix_Span TopRight(int rows, int cols) const { return Matrix_Span(rows, lda, n - cols, data, 0, cols); }
  // Bottom rows | Left cols
  Matrix_Span BotLeft(int rows, int cols) const { return Matrix_Span(m - rows, lda, cols, data, rows, 0); }
  // Bottom rows | Right cols
  Matrix_Span BotRight(int rows, int cols) const { return Matrix_Span(m - rows, lda, n - cols, data, rows, cols); }

  // Exercise 53.8 METHODS End

  // Exercise 53.9 METHODS Start

  // General Matrix Multiplication
  void MatMult(Matrix_Span& other, Matrix_Span& out) const {
    timeExecution("MatMult", [&](Matrix_Span& otherRef, Matrix_Span& outRef) {
      if (n != otherRef.m || m != outRef.m || otherRef.n != outRef.n) {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
      }

      for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < otherRef.n; ++j) {
          for (auto k = 0; k < n; ++k) {
            outRef.at(i, j) += this->at(i, k) * otherRef.at(k, j);
          }
        }
      }
    }, other, out);
  }

  // Blocked Matrix Multiplication
  void BlockedMatMult(Matrix_Span& other, Matrix_Span& out) const {
    timeExecution("BlockedMatMult", [&](Matrix_Span& otherRef, Matrix_Span& outRef) {
      if (n != otherRef.m || m != outRef.m || otherRef.n != outRef.n) {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
      }

      int mid = m / 2; // Assuming square matrix divisible by 2

      // Top-left
      Matrix_Span A11 = this->TopLeft(mid, mid);
      Matrix_Span B11 = otherRef.TopLeft(mid, mid);
      Matrix_Span C11 = outRef.TopLeft(mid, mid);

      // Top-right
      Matrix_Span A12 = this->TopRight(mid, mid);
      Matrix_Span B21 = otherRef.BotLeft(mid, mid);
      Matrix_Span C12 = outRef.TopRight(mid, mid);

      // Bottom-left
      Matrix_Span A21 = this->BotLeft(mid, mid);
      Matrix_Span B12 = otherRef.TopRight(mid, mid);
      Matrix_Span C21 = outRef.BotLeft(mid, mid);

      // Bottom-right
      Matrix_Span A22 = this->BotRight(mid, mid);
      Matrix_Span B22 = otherRef.BotRight(mid, mid);
      Matrix_Span C22 = outRef.BotRight(mid, mid);

      // Recursive Computation
      A11.RecursiveMatMult(B11, C11);
      A12.RecursiveMatMult(B21, C11);

      A11.RecursiveMatMult(B12, C12);
      A12.RecursiveMatMult(B22, C12);

      A21.RecursiveMatMult(B11, C21);
      A22.RecursiveMatMult(B21, C21);

      A21.RecursiveMatMult(B12, C22);
      A22.RecursiveMatMult(B22, C22);
    }, other, out);
  }

  // Recursive Matrix Multiplication
  void RecursiveMatMult(Matrix_Span& other, Matrix_Span& out) const {
    timeExecution("RecursiveMatMult", [&](Matrix_Span& otherRef, Matrix_Span& outRef) {
      if (m == 1 && n == 1 && otherRef.n == 1) {
        outRef.at(0, 0) += this->at(0, 0) * otherRef.at(0, 0);
        return;
      }

      if (m <= 2 || n <= 2 || otherRef.n <= 2) {
        this->MatMult(otherRef, outRef);
        return;
      }

      // Midpoint for division
      int midM = m / 2 + (m % 2); // Padding for odd dimensions
      int midN = n / 2 + (m % 2);
      int midP = other.n / 2 + (other.n % 2);

      // Divide matrices into submatrices
      auto A11 = this->TopLeft(midM, midN);
      auto A12 = this->TopRight(midM, midN);
      auto A21 = this->BotLeft(midM, midN);
      auto A22 = this->BotRight(midM, midN);

      auto B11 = other.TopLeft(midN, midP);
      auto B12 = other.TopRight(midN, midP);
      auto B21 = other.BotLeft(midN, midP);
      auto B22 = other.BotRight(midN, midP);

      auto C11 = out.TopLeft(midM, midP);
      auto C12 = out.TopRight(midM, midP);
      auto C21 = out.BotLeft(midM, midP);
      auto C22 = out.BotRight(midM, midP);

      // Temporary matrices to store intermediate results
      vector<double> temp1_data(midM * midP, 0.0);
      vector<double> temp2_data(midM * midP, 0.0);
      Matrix_Span temp1(midM, midM, midP, temp1_data.data());
      Matrix_Span temp2(midM, midM, midP, temp2_data.data());

      // Compute the 2x2 block matrix multiplication
      A11.RecursiveMatMult(B11, temp1);
      A12.RecursiveMatMult(B21, temp2);
      C11.def_addMatrices(temp1, temp2);

      A11.RecursiveMatMult(B12, temp1);
      A12.RecursiveMatMult(B22, temp2);
      C12.def_addMatrices(temp1, temp2);

      A21.RecursiveMatMult(B11, temp1);
      A22.RecursiveMatMult(B21, temp2);
      C21.def_addMatrices(temp1, temp2);

      A21.RecursiveMatMult(B12, temp1);
      A22.RecursiveMatMult(B22, temp2);
      C22.def_addMatrices(temp1, temp2);
    }, other, out);
  }

  // Exercise 53.9 METHODS End
};

  // Exercise 53.5 Matrix_Mdspan Class
// COME BACK TO LATER TO FIGURE OUT MDSPAN
/**
  class Matrix_Mdspan {
private:
  int m, lda, n; // Rows (m), Columns (n), Leading Dimension of A (lda)
    mdspan<double, dextents<int, 2>> data; // Non-owning view of data
public:
  // Constructer for top-level matrix
  Matrix_Mdspan(int m, int lda, int n, double* ptr)
    : m(m), lda(lda), n(n), data(ptr, lda, n)
  {}

  // Constructor for submatrix
    Matrix_Mdspan(int m, int lda, int n, mdspan<double, dextents<int, 2>> parent, int rowOffset, int colOffset)
      : m(m), lda(lda), n(n), data(parent.accessor().data() + colOffset * lda + rowOffset, lda * n)
  {}

  // Access matrix elements
  double& operator()(int i, int j) {
    if (i >= m || j >= n) throw std::out_of_range("Index out of bounds");
    return data(i, j);
  }

  const double& operator()(int i, int j) const {
    if (i >= m || j >= n) throw std::out_of_range("Index out of bounds");
    return data(i, j);
  }

  // Print matrix
  void print() const {
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        cout << (*this)(i, j) << " ";
      }
      cout << '\n';
    }
  }

  double& at (int i, int j) {
    return data(i, j);
  }
};
**/

int main() {

  // Exercise 53.1 CODE Start

  // Define matrices using vectors
  vector<vector<int>> a = {{1, 2, 3}, {4, 5, 6}};
  vector<vector<int>> b = {{1, 2}, {3, 4}, {5, 6}};
  vector<vector<int>> c(a.size(), vector<int>(b[0].size(), 0)); // Initialize result matrix with zeros
  vector<vector<int>> d(a.size(), vector<int>(b[0].size(), 0)); // Initialize result matrix with zeros

  // Perform matrix multiplication
  for (int i = 0; i < a.size(); i++) { // Rows of a
      for (int j = 0; j < b[0].size(); j++) { // Columns of b
          for (int k = 0; k < a[0].size(); k++) { // Columns of a / Rows of b
              c[i][j] += a[i][k] * b[k][j];
          }
      }
  }

  // Output result
  cout << "Straightforward Output" << '\n';
  for (int i = 0; i < c.size(); i++) {
      for (int j = 0; j < c[0].size(); j++) {
          cout << c[i][j] << " ";
      }
      cout << "\n";
  }

  // Permuted Algorithm to Perform matrix multiplication
  for (int j = 0; j < b[0].size(); j++) { // Columns of b
        for (int i = 0; i < a.size(); i++) { // Rows of a
            for (int k = 0; k < a[0].size(); k++) { // Columns of a / Rows of b
                d[i][j] += a[i][k] * b[k][j];
            }
        }
  }

  // Permuted Algorithm Output result
  cout << "\nPermuted Algorithm Output:" << '\n';
  for (int i = 0; i < d.size(); i++) {
      for (int j = 0; j < d[0].size(); j++) {
          cout << d[i][j] << " ";
      }
      cout << "\n";
  }

  // Exercise 53.1 CODE End

  // Exercise 53.2 CODE Start

  Matrix A({{1, 2, 3}, {4, 5, 6}});
  Matrix B({{1, 2}, {3, 4}, {5, 6}});

  cout << "\nMatrix A:\n";
  A.print();

  cout << "\nMatrix B:\n";
  B.print();

  cout << "\nTraditional Matrix-Matrix Multiplication (A * B):\n";
  Matrix C = A.MatMult(B);
  C.print();

  cout << "\nRecursive Matrix-Matrix Multiplication (A * B):\n";
  Matrix D = A.MatMultRecur(B);
  D.print();

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

  // Create submatrix
  Matrix_Span sub(m, lda, 2, span<double>(data_1), 0, 1);

  // Print submatrix
  cout << "\nSubmatrix:\n";
  sub.print();

  // Modify element within submatrix
  sub.at(1, 0) = 99;

  // Print updated submatrix
  cout << "\nUpdated Submatrix:\n";
  sub.print();

  // Print updated top-level matrix
  cout << "\nUpdated Top-Level Matrix\n";
  one.print();

  // Print out value at i = 0 and j = 1 for top-level matrix
  cout << "\nTop-Level Matrix (0, 1): " << one.at(0, 1);

  // Print out value at i = 1 and j = 0 for submatrix
  cout << "\nSubmatrix (0, 1): " << sub.at(1, 0) << '\n';

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
  Matrix_Span four = two.addMatrices(three, result_data, result);

  // Print added matrix
  cout <<"\nAdded Matrix (Two + Three):\n";
  four.print();

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
  C_def.def_addMatrices(A_def, B_def);

  // Print Results
  cout << "\nAdded Matrix using Def (C_def = A_def + B_def):\n";
  C_def.def_print();

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
  cout << "\nLeft(3):\n";
  Matrix_Span left = S.Left(3);
  left.print();

  cout << "\nRight(3):\n";
  Matrix_Span right = S.Right(3);
  right.print();

  cout << "\nTop(2):\n";
  Matrix_Span top = S.Top(2);
  top.print();

  cout << "\nBot(2):\n";
  Matrix_Span bot = S.Bot(2);
  bot.print();

  cout << "\nTopLeft(2, 3):\n";
  Matrix_Span topLeft = S.TopLeft(2, 3);
  topLeft.print();

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
  vector<double> data_mult2(lda_mult * n_mult, 1.0);
  vector<double> temp_mult1(lda_mult * n_mult, 0.0);
  vector<double> temp_mult2(lda_mult * n_mult, 0.0);
  for (auto i = 0; i < m_sub; ++i) {
    for (auto j = 0; j < n_sub; ++j) {
      data_mult1[j * lda_sub + i] = i + j * 0.1; // Initialize with unique values
      data_mult2[j * lda_sub + i] = i + j * 0.1; // Initialize with unique values
    }
  }

  // Create Top-Level Matrices
  Matrix_Span A_mult(m_mult, lda_mult, n_mult, data_mult1.data());
  Matrix_Span B_mult(m_mult, lda_mult, n_mult, data_mult2.data());
  Matrix_Span D_mult(m_mult, lda_mult, n_mult, temp_mult1.data());
  Matrix_Span E_mult(m_mult, lda_mult, n_mult, temp_mult2.data());

  for (auto i = 0; i < m_mult; ++i) {
    for (auto j = 0; j < n_mult; ++j) {
      D_mult.at(i, j) = 0.0;
    }
  }

  // Print Matrices
  cout << "\nMatrix A_mult:\n";
  A_mult.def_print();

  cout << "\nMatrix B_mult:\n";
  B_mult.def_print();

  cout << "\nMatrix D_mult:\n";
  D_mult.def_print();

  cout << "\nMatrix E_mult:\n";
  E_mult.def_print();

  // Perform Traditional Matrix Multiplication
  A_mult.MatMult(B_mult, D_mult);

  // Perform Recursive Blocked Matrix Multiplication
  A_mult.BlockedMatMult(B_mult, E_mult);

  // Print Resulting Matrix
  cout << "\nResulting Matrix D_mult (A_mult * B_mult):\n";
  D_mult.def_print();

  cout << "\nResulting Matrix E_mult (A_mult * B_mult):\n";
  E_mult.def_print();

  return 0;
}
