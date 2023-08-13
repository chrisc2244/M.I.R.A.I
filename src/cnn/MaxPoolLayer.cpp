#include "MaxPoolLayer.h"
#include <iostream>
#include "../utils/Log.h"

MaxPoolLayer::MaxPoolLayer(const std::string& label, uint8_t stride, bool rightLoss)
	: Layer(label), m_stride(stride), m_inputTensor(nullptr), m_outputTensor(nullptr), m_rightLoss(rightLoss), m_currentTensor(0)
{}

MaxPoolLayer::~MaxPoolLayer()
{
	delete m_outputTensor;
}

void MaxPoolLayer::init(TensorPtrs* inputTensor)
{
	m_inputTensor = inputTensor;
}

bool MaxPoolLayer::isDone()
{
	return m_isDone;
}

void MaxPoolLayer::step()
{
	uint16_t cols, rows;
	// Loop through every matrix in the tensor
	for (; m_currentTensor < m_inputTensor->size(); m_currentTensor++)
	{
		// Number of rows an columns in each matrix for the INPUT tensor
		cols = m_inputTensor->getElement(m_currentTensor)->getColumnAmount();
		rows = m_inputTensor->getElement(m_currentTensor)->getRowAmount();

		// Number of rows an columns in each matrix for the OUPUT tensor
		uint16_t outCols = cols / m_stride;
		uint16_t outRows = rows / m_stride;

		// Initialize the output tensor if not yet initialized
		if (!m_outputTensor) {
			m_outputTensor = new Tensor(m_inputTensor->size());
		}

		// Initialize the output matrix for this input matrix
		// The size of each output matrix is decreased by a FACTOR of stride
		m_outputTensor->addElement(Matrix(outRows, outCols));

		// Starting row and column for pooling
		uint16_t startC{ 0 }, 
				 startR{ 0 };
		
		// Loop through every stride size element in the matrix
		if (m_rightLoss) // If the data loss should be stride amount on right/bottom
		{
			uint16_t out_i = 0;
			uint16_t out_j = 0;
			for (uint16_t i = startC; i < cols; i += m_stride)
			{
				out_j = 0;
				for (uint16_t j = startR; j < rows; j += m_stride)
				{
					// Get max element
					double max = 0.0f;
					for (uint8_t x = 0; x < m_stride; x++)
					{
						for (uint8_t y = 0; y < m_stride; y++)
						{
							if ((*m_inputTensor->getElement(m_currentTensor))(i + x, j + y) > max)
								max = (*m_inputTensor->getElement(m_currentTensor))(i + x, j + y);
						}
					}
					(*m_outputTensor->getElement(m_currentTensor))(out_i, out_j) = max;
					out_j++;
				}
				out_i++;
			}
		}
		else // If the data loss should be stride amount on left/top
		{
			// Get the starting point
			startC = cols % m_stride;
			startR = rows % m_stride;

			uint16_t out_i = 0;
			uint16_t out_j = 0;
			for (uint16_t i = startC; i < cols; i += m_stride)
			{
				out_j = 0;
				for (uint16_t j = startR; j < rows; j += m_stride)
				{
					// Get max element
					double max = 0.0f;
					for (uint8_t x = 0; x < m_stride; x++)
					{
						for (uint8_t y = 0; y < m_stride; y++)
						{
							if ((*m_inputTensor->getElement(m_currentTensor))(i + x, j + y) > max)
								max = (*m_inputTensor->getElement(m_currentTensor))(i + x, j + y);
						}
					}
					(*m_outputTensor->getElement(m_currentTensor))(out_i, out_j) = max;
					out_j++;
				}
				out_i++;
			}
		}
	}
	m_isDone = true;
	// #DEBUG dumps the first matrix in the output tensor to file in project root directory with name of the layer
	Matrix::dumpToFile(m_Id+".txt", *m_outputTensor->getElement(0));
	MIR::Log::writefInfo("MaxPoolLayer::step()", "Layer \"%s\" finished processing...", m_Id.c_str());
}