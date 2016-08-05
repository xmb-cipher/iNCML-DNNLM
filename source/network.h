/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : network.h
Last Update : Feb 2, 2016
Description : Provide interface of a general-purpose neural network
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */


#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "layer.h"
#include <fstream>
#include <string>

class Network {
protected:
    vector<Layer*>            m_layer;
    vector<Matrix*>           m_gradient;
    vector<const MatrixBase*> m_input;
    vector<Matrix*>           m_output;
    vector<int>                    m_input_size;
    vector<int>                    m_output_size;

public:
    Network( const char* config );
    ~Network();
    const MatrixBase& Compute( const MatrixBase& feature );
    void BackPropagate( const MatrixBase& target, float lr, MatrixBase* gradient = NULL );
    void SaveParam( const char* filename );
    void LoadParam( const char* filename );
    void SaveTrainingBuffer( const char* filename );
    void LoadTrainingBuffer( const char* filename );
    void Prepare( const ExtraInfo& info );
    void Report();
    int InputSize() const {
        return m_input_size.front();
    }
    int OutputSize() const {
        return m_output_size.back();
    }
};

#endif
