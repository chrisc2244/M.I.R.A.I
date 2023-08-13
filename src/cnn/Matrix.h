#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
class Matrix
{

public:

	//constructors
	Matrix();

	// Copy Constructor
	Matrix(const Matrix& m);

	/**
	* Creates a matrix of columns x rows size, and populates it with values from a vector
	*
	* @param rows - The row amount (height) of the matrix
	* @param columns - The column amount (width) of the matrix
	* @param values -  Reference to a vector containing the values to populate the matrix with
	*/
	Matrix(const uint16_t rows, const uint16_t columns, const std::vector<double> &values);

	/**
	* Creates a matrix of columns x rows size, and populates it with values from a vector
	*
	* @param rows - The row amount (height) of the matrix
	* @param columns - The column amount (width) of the matrix
	* @param arrPtr -  Pointer to a heap allocated array of doubles for the matrix to use
	*/
	Matrix(const uint16_t rows, const uint16_t columns, double* arrPtr);

	/**
	* Creates a matrix of columns x rows size, and populates it with default_value
	*
	* @param rows - The row amount (height) of the matrix
	* @param columns - The column amount (width) of the matrix
	* @param default_value - Value to place in every element of the matrix
	*/
	Matrix(const uint16_t rows, const uint16_t columns, double default_value = 0);


	/**
	* Creates a matrix of columns x rows size, and populates it with random value
	* If random is set to false, it will populate the matrix with the low bound value
	* @param rows - The row amount (height) of the matrix
	* @param columns - The column amount (width) of the matrix
	* @param random - Flag to populate the with random values between low and high
	* @param low - Minimum value for the random value  (INCLUSIVE)
	* @param high - Maximum value for the random value (INCLUSIVE)
	*/
	Matrix(const uint16_t rows, const uint16_t columns, bool random, float low = 0.0f, float high = 1.0f);

	~Matrix();

	/**
	* Places a submatrix of the CALLING object into the destination matrix, note that destination matrix
	* must have a width and height in columns and rows for this function to work.
	*
	* @param startcol - The column (x index) of the first element to put in the submatrix
	* @param startrow - The row (y index) of the first element to put in the submatrix
	* @param destination - A reference to an intialized matrix in which the put the submatrix
	*
	* @throws out_of_bounds exceptions if startcol, startrow, or that index + destination columns and rows is out of bounds.
	*/
	void putSubMatrix(uint16_t startcol, uint16_t startrow, Matrix& destination);

	/**
	* Fills the matrix with the values in values vector
	*
	* It will fill the entire matrix if values is greater than or equal to the number
	* of elements in this matrix. It will NOT resize to accomadate extra data or shrink to fit
	* a smaller data set
	*/
	void setTo(const std::vector<double>& values);

	/**
	* Recreates the matrix of columns x rows size, and populates it with default_value
	*
	* @param rows - The row amount (height) of the matrix
	* @param columns - The column amount (width) of the matrix
	* @param default_value - Value to place in every element of the matrix
	*/
	void reset(uint16_t cols, uint16_t rows, double default_value = 0.0f);
	
	// Returns number of rows (height)
	inline uint16_t getRowAmount() const { return m_rows; }

	// Returns number of columns (width)
	inline uint16_t getColumnAmount() const { return m_columns;  }

	// Returns the size of the matrix in elements (columns * rows)
	inline uint16_t getSize() const { return m_size; }
	
	//multiplies two matrices together index by index
	//ex: (matrix[1] * matrix[1]) then divide by the total sum of pixels (sum / m_size)
	//essentially the result of applying the filter 
	double mutliplyMatricesIndexByIndexThenDivideBySize(const Matrix& matrixToMultiply);

	//returns matrix of specified window. Remember, (0,0) is first element
	//so getSubMatrix(3, 3, 2, 2) returns 2x2 window starting at (kinda) 4th row, 4th column
	
	//same as below but returns a filled vector instead of a new matrix object
	std::vector<double> getSubMatrixVector(uint16_t start_row, uint16_t start_col, uint16_t window_rows, uint16_t window_cols);

	//returns matrix of specified window. Remember, (0,0) is first element
	//so getSubMatrix(3, 3, 2, 2) returns 2x2 window starting at (kinda) 4th row, 4th column
	//ignore this, I want it later
	Matrix getSubMatrix(uint16_t start_row, uint16_t start_col, uint16_t window_rows, uint16_t window_cols);

	//prints values in Matrix		[0.4213][0.4213][0.4213]
	//3x3 example, 6 values total	[0.4213][0.4213][0.4213]
	static void print(const Matrix& MatrixToPrint);
	
	/**
	* Generates a string that represents the matrix and its elements
	*
	* @param matrix - A reference to the matrix you want a string representation of
	*/
	static std::string toString(const Matrix& matrix);

	static bool dumpToFile(std::string filename, const Matrix& mat);

	//operators
	Matrix operator+(const Matrix& otherMatrix) const;
	Matrix operator-(const Matrix& otherMatrix) const;
	Matrix operator*(const Matrix& otherMatrix) const;		// Dot Product
	void operator=(const Matrix& otherMatrix);						
	double& operator[](uint32_t index) const;			// Access element in Matrix[index]
	double& operator() (uint16_t row, uint16_t col) const;		// Access element in Matrix in (row, col) form (y, x) starting at (0,0) for first element

private:

	uint16_t m_rows;
	uint16_t m_columns;
	uint32_t m_size;
	double* m_ptr_double;
};
