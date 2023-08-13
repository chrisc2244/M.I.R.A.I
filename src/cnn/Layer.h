#pragma once
#include "Node.h"
#include <vector>
#include "TensorPtrs.h"


class Layer
{
public:

	Layer(std::string id = "");

	virtual void step() = 0;

	// Returns trus if layer is done processing
	virtual bool isDone() = 0;

	inline const char* getId() const { return m_Id.c_str(); }

	double static reLU();
	double static sigmoid();
	
	
protected:
	bool m_isDone; // Flag for whether or not the layer has finished processing

	std::string m_Id;
};
