#pragma once
#include <vector>
#include "Matrix.h"
#include "Tensor.h"

class Node
{

public:
	Node();
	Node(Matrix* filter, int bias);
	Node(Matrix* filter);

	// Node takes ownership of it's filter, deletes the memory here
	~Node();


	Matrix* getWeights() const;
	void setBias(uint8_t bias);

	void setFilter(Matrix* weights);
	double getProcessedResult();

	const inline int getBias() const { return m_bias; }

	// Update 1/23/23

	// Sets the size of the output matrices in the tensor
	void initOutputs(uint16_t amount, uint16_t rows, uint16_t cols);
	void applyFilter(Matrix& currentWindow, uint16_t currentInput, uint16_t column, uint16_t row);

	// Combines every matrix in the output tensor into a single output matrix
	void combineOutputs();

	inline Matrix* getOutput() const { return m_CombinedOutput; }


private:

	Matrix* m_filter; // Put this on the heap, give ownership to node
	uint8_t m_bias;
	double m_processedResult;

	// Update 1/23/23 
	Tensor* m_Outputs;

	// Combination of every matrix in the output tensor in a single output matrix
	Matrix* m_CombinedOutput;
};