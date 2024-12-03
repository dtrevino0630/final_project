#include <iostream>
using std::cout;
using namespace std;

#include <vector>
using std::vector;

#include <cmath>

int main() {
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
  cout << "Permuted Algorithm Output:" << '\n';
  for (int i = 0; i < d.size(); i++) {
      for (int j = 0; j < d[0].size(); j++) {
          cout << d[i][j] << " ";
      }
      cout << "\n";
  }
  return 0;
}

