/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : batch-constructor.h
Last Update : Feb 2, 2016
Description : Provide interfaces to construct mini-batch for SGD in language modelling
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

#ifndef BATCH_CONSTRUCTOR_H_INCLUDED
#define BATCH_CONSTRUCTOR_H_INCLUDED

#include "matrix.h"
#include <algorithm>
#include <fstream>
#include <algorithm>

class BatchConstructor {
protected:
    Matrix                                 m_gpu_buffer;
    vector<float>                        m_cpu_buffer;
    vector<vector<float> >                m_sentence;
    vector<vector<float> >::iterator    m_itr;
    int                                  m_n_sample;
    int                                 m_batch_size;
    int                                 m_n_order;
    float                                m_start_idx;
    int                                    m_n_sentence_discarded;
    int                                    m_n_example_discarded;
    bool                                m_is_training;

    SubMatrix                            m_input;
    SubMatrix                            m_target;
    SubMatrix                            m_noise;
    SubMatrix                            m_target_and_noise;
    SubMatrix                            m_length;
    SubMatrix                            m_position;


public:
    BatchConstructor( const char* filename, 
                      int order, 
                      int batchSize, 
                      int sampleSize, 
                      bool isTrainMode = true );
    ~BatchConstructor();
    void Reset();
    bool HasNext() const;
    void PrepareNext( const MatrixBase* accUnigram = NULL, const float* wordCount = NULL );
    SubMatrix GetInput();
    SubMatrix GetTarget();
    SubMatrix GetNoise();
    SubMatrix GetTargetAndNoise();
    SubMatrix GetSentenceLength();
    SubMatrix GetPosition();
};

#endif
