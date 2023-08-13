#include "Node.h"
#include <iostream>
#include "Matrix.h"



Node::Node()
	: m_filter(nullptr), m_bias(0), m_processedResult(0), m_Outputs(nullptr), m_CombinedOutput(nullptr)
{
}

Node::Node(Matrix* filter, int bias = 0)
	: m_processedResult(0), m_Outputs(nullptr), m_CombinedOutput(nullptr)
{
	m_filter = filter;
	m_bias = bias;
}

Node::Node(Matrix* filter) 
	: m_bias(0), m_processedResult(0), m_Outputs(nullptr)
{
	m_filter = filter;
}

Node::~Node()
{
	delete m_filter;
	delete m_Outputs;
	delete m_CombinedOutput;
}

Matrix* Node::getWeights() const
{
	return m_filter;
}


void Node::setBias(const uint8_t bias)
{
	m_bias = bias;
}

void Node::setFilter(Matrix* weights)
{
	m_filter = weights;
}

double Node::getProcessedResult()
{
	return m_processedResult;
}

void Node::initOutputs(uint16_t amount, uint16_t rows, uint16_t cols)
{
	m_Outputs = new Tensor(amount);

	// Add amount number of matrices of cols x rows size to the tensor
	for (uint16_t i = 0; i < amount; i++)
	{
		m_Outputs->addElement(Matrix(rows, cols));
	}
}

void Node::applyFilter(Matrix& currentWindow, uint16_t currentInput, uint16_t column, uint16_t row)
{
	double valueOfFilterAtCurrentWindow = currentWindow.mutliplyMatricesIndexByIndexThenDivideBySize(*m_filter);
	m_processedResult = valueOfFilterAtCurrentWindow;
	(*(*m_Outputs)[currentInput])(row, column) = m_processedResult;
}

void Node::combineOutputs()
{

	uint16_t rows = m_Outputs->getElement(0)->getRowAmount();
	uint16_t cols = m_Outputs->getElement(0)->getColumnAmount();

	uint8_t size = m_Outputs->size();

	m_CombinedOutput = new Matrix(rows, cols);

	// Sum of the values in each output matrix
	double elementSum = 0.0f;

	// Loop through every element in matrix 
	for (uint16_t r = 0; r < rows; r++)
	{
		for (uint16_t c = 0; c < cols; c++)
		{
			elementSum = 0.0f;
			// Loop through each vector and keep track of
			for (Matrix& m : m_Outputs->get())
			{
				elementSum += m(r, c);
			}
			elementSum /= size;

			(*m_CombinedOutput)(r, c) = elementSum;
		}
	}

}
