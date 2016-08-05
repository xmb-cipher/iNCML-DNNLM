/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : layer.h
Last Update : Feb 2, 2016
Description : Provide interfaces of various neural network layers
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */


#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "matrix.h"
#include <fstream>

struct ExtraInfo {
    int rank;                    // fofe & sfsmn
    SubMatrix length;            // fofe & sfsmn
    bool nce;                    // nce
    SubMatrix targetAndNoise;    // nce
    const Matrix& unigram;        // nce
    SubMatrix position;            // vfsmn
    ExtraInfo( int r, SubMatrix l, bool n, SubMatrix tan, const Matrix& u, SubMatrix p )
        : rank(r), length(l), nce(n), targetAndNoise(tan), unigram(u), position(p) {}
    ExtraInfo() : unigram( Matrix() ) {}
};


class Layer {
protected:
    Layer() {};

public:
    virtual ~Layer() {}
    virtual void Prepare( const ExtraInfo& info ) {}
    virtual void Compute( const MatrixBase& feature, MatrixBase* output ){}
    virtual void BackPropagate( MatrixBase& outDiff,     // may be used as buffer, thus its content may not preserved
                                const MatrixBase& in,
                                const MatrixBase& out,
                                float learningRate,
                                MatrixBase* inDiff ){}
    virtual void SaveParam( ofstream& ofs ) {}
    virtual void LoadParam( ifstream& ifs ) {}
    virtual void SaveTrainingBuffer( ofstream& ofs ) {}
    virtual void LoadTrainingBuffer( ifstream& ifs ) {}
    virtual string Report() { return "<Layer> has nothing to report"; }
};

// =================================================================================

class ProjectionLayer : public Layer {
protected:
    Matrix m_projection;
    Matrix m_gradient;    // not used
    float m_momentum;
    float m_weight_decay;
    string m_norm_type;
    float m_norm_param;
    string m_report();

public:
    friend class FofeLayer;
    ProjectionLayer( int n_vocab, int n_projection, 
                     float momentum = 0.0f, float weight_decay = 0.0f, 
                     string norm_type = "", float norm_param = 10.0f );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff = NULL );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    string Report();
};

// =================================================================================

class FofeLayer : public Layer {
protected:
    ProjectionLayer m_projection;
    Matrix m_block_diagonal;
    Matrix m_buffer;
    float m_alpha;

public:
    FofeLayer( int n_vocab, int n_projection, 
               float alpha = 0.7f, float momentum = 0.0f, float weight_decay = 0.0f,
               string norm_type = "", float norm_param = 10.0f );
    void Prepare( const ExtraInfo& info );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff = NULL );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    string Report();
};

// =================================================================================

class LinearLayer : public Layer {
protected:
    Matrix m_weight;
    Matrix m_bias;
    Matrix m_w_momentum;
    Matrix m_b_momentum;
    float m_momentum;
    float m_weight_decay;
    string m_norm_type;
    float m_norm_param;
    bool m_has_bias;
    string m_report();

public:
    friend class ReLULayer;
    friend class SigmoidLayer;
    friend class SoftmaxLayer;
    friend class NCELayer;
    friend class sFSMNLayer;
    friend class vFSMNLayer;

    LinearLayer( int n_in, int n_out, 
                 float momentum = 0.0f, float weight_decay = 0.0f,
                 string norm_type = "", float norm_param = 10.0f,
                 bool has_bias = true );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void Compute( const SparseMatrix& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff );
    void BackPropagate( MatrixBase& outDiff,
                        const SparseMatrix& in,    // since the input is sparse
                        const MatrixBase& out,
                        float learningRate );    // we don't need its gradient
    void SetWeights( const vector<float>& weights );
    void GetWeights( vector<float>* weights );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    void SaveTrainingBuffer( ofstream& ofs );
    void LoadTrainingBuffer( ifstream& ifs );
    string Report();
};

// =================================================================================

class sFSMNLayer : public Layer {
protected:
    LinearLayer m_linear;
    Matrix m_filter;
    Matrix m_weight;
    Matrix m_w_momentum;
    Matrix m_memory;
    Matrix m_memory_diff;
    Matrix m_block_diagonal;
    Matrix m_diagonal_diff;
    Matrix m_length;
    float m_momentum;
    float m_weight_decay;
    string m_norm_type;
    float m_norm_param;
    
public:
    sFSMNLayer( int n_in, int n_out, int n_order = 20, 
                float momentum = 0.0f, float weight_decay = 0.0f,
                string norm_type = "", float norm_param = 10.0f );
    void Prepare( const ExtraInfo& info );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    void SaveTrainingBuffer( ofstream& ofs );
    void LoadTrainingBuffer( ifstream& ifs );
    string Report();
};

// =================================================================================


class vFSMNLayer : public Layer {
protected:
    LinearLayer m_linear;
    Matrix m_weight;
    Matrix m_filter;
    Matrix m_memory;
    Matrix m_position;
    Matrix m_w_momentum;
    Matrix m_memory_diff;
    float m_momentum;
    float m_weight_decay;
    string m_norm_type;
    float m_norm_param;

public:
    vFSMNLayer( int n_in, int n_out, int n_order = 20, 
                float momentum = 0.0f, float weight_decay = 0.0f,
                string norm_type = "", float norm_param = 10.0f );
    void Prepare( const ExtraInfo& info );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    void SaveTrainingBuffer( ofstream& ofs );
    void LoadTrainingBuffer( ifstream& ifs );
    string Report();
};

// =================================================================================

class ReLULayer : public Layer {
protected:
    LinearLayer m_linear;

public:
    ReLULayer( int n_in, int n_out, 
               float momentum = 0.0f, float weight_decay = 0.0f,
               string norm_type = "", float norm_param = 10.0f );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    void SaveTrainingBuffer( ofstream& ofs );
    void LoadTrainingBuffer( ifstream& ifs );
    string Report();
};

// =================================================================================

class SigmoidLayer : public Layer {
protected:
    LinearLayer m_linear;

public:
    SigmoidLayer( int n_in, int n_out, 
                  float momentum = 0.0f, float weight_decay = 0.0f,
                  string norm_type = "", float norm_param = 10.0f );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& outDiff, 
                        const MatrixBase& in,
                        const MatrixBase& out,
                        float learningRate,
                        MatrixBase* inDiff );
    void SaveParam( ofstream& ofs );
    void LoadParam( ifstream& ifs );
    void SaveTrainingBuffer( ofstream& ofs );
    void LoadTrainingBuffer( ifstream& ifs );
    string Report();
};

// =================================================================================

class SoftmaxLayer : public Layer {
protected:
    LinearLayer m_linear;

public:
    friend class NCELayer;

    SoftmaxLayer( int n_in, int n_out, 
                  float momentum = 0.0f, float weight_decay = 0.0f,
                  string norm_type = "", float norm_param = 10.0f );
    virtual void Compute( const MatrixBase& feature, MatrixBase* output );
    virtual void BackPropagate( MatrixBase& out, 
                                const MatrixBase& in,
                                const MatrixBase& target,
                                float learningRate,
                                MatrixBase* inDiff );
    virtual void SaveParam( ofstream& ofs );
    virtual void LoadParam( ifstream& ifs );
    virtual void SaveTrainingBuffer( ofstream& ofs );
    virtual void LoadTrainingBuffer( ifstream& ifs );
    string Report();
};

// =================================================================================

class NCELayer : public SoftmaxLayer {
protected:
    bool m_nce_enabled;
    float m_lnZ;
    Matrix m_target_and_noise;
    Matrix m_partial_weight;
    Matrix m_partial_bias;
    Matrix m_partial_w_gradient;
    Matrix m_partial_b_gradient;
    Matrix m_unigram;
    string m_norm_type;
    float m_norm_param;

public:
    NCELayer( int n_in, int n_out, float lnZ, 
              string norm_type = "", float norm_param = 10.0f );
    void Prepare( const ExtraInfo& info );
    void Compute( const MatrixBase& feature, MatrixBase* output );
    void BackPropagate( MatrixBase& out, 
                        const MatrixBase& in,
                        const MatrixBase& target,
                        float learningRate,
                        MatrixBase* inDiff );
};

#endif
