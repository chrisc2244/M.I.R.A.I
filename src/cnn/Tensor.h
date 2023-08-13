#pragma once
#include <vector>	
#include "Matrix.h"

class Tensor
{
public:
	Tensor();
	Tensor(uint8_t size);
	void addElement(const Matrix& m);
	void setElement(uint8_t index, const Matrix& m);
	inline Matrix* getElement(uint8_t index) { return &m_Matrices[index]; }
	const inline uint8_t size() const { return m_NumElements; }

	inline std::vector<Matrix>& get() { return m_Matrices; }

	Matrix* operator[](uint8_t index);
private:
	std::vector<Matrix> m_Matrices;
	uint8_t m_NumElements;
};