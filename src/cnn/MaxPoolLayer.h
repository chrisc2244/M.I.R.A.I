#pragma once
#include "Layer.h"
#include <string>

class MaxPoolLayer : virtual public Layer
{
public:
	/**
	* Create a Max Pooling layer
	* @param label - String label for the layer
	* @param strideX - amount of columns to combine into one output
	* @param strideY - amount of rows to combine into one output
	* @param rightLoss - Whether or not the data loss is right/bottom or left/top
	*/
	MaxPoolLayer(const std::string& label, uint8_t stride, bool rightLoss);

	~MaxPoolLayer();

	/**
	* Initialize the pooling layer with the a tensor of input matricies
	*/
	void init(TensorPtrs* inputTensor);

	void step();

	bool isDone();

private:
	// Input for the pooling layer
	TensorPtrs* m_inputTensor;
	// Pooled data output
	Tensor* m_outputTensor;

	// Size of the block that gets pooled
	uint8_t m_stride;

	// Index of the current element of the tensor being pooled
	uint8_t m_currentTensor;

	// Whether or not the data loss is right/bottom or left/top
	bool m_rightLoss;
};