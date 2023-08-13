#include "Tensor.h"

Tensor::Tensor()
	: m_NumElements(0)
{
}

Tensor::Tensor(uint8_t size)
	: m_NumElements(size)
{
}

void Tensor::addElement(const Matrix& m)
{
	m_Matrices.emplace_back(m);
	m_NumElements++;
}

void Tensor::setElement(uint8_t index, const Matrix& m)
{
	m_Matrices[index] = m;
}

Matrix* Tensor::operator[](uint8_t index)
{
	return &(m_Matrices[index]);
}
