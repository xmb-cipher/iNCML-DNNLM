/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : layer.cpp
Last Update : Feb 2, 2016
Description : Provide concrete implementation of various neural network layers
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -o main -lcublas -lcurand -Xcompiler -rdynamic -DLAYER_UNIT_TEST matrix.cu layer.cpp

#include "layer.h"

// =================================================================================

sFSMNLayer::sFSMNLayer( int n_in, int n_out, int n_order, 
                        float momentum, float weight_decay,
                        string norm_type, float norm_param )
    :    m_linear( n_in, n_out, momentum, weight_decay, norm_type, norm_param ),
        m_filter( 1, n_order + 1 ),
        m_weight( n_in, n_out ),
        m_momentum( momentum ),
        m_weight_decay( weight_decay ),
        m_norm_type( norm_type ),
        m_norm_param( norm_param ) {
    // float param_range = 1.0f / sqrt( (float)n_in );
    float param_range = sqrt( 6.0f / (float)(n_in + n_out) );
    m_weight.Random( -param_range, param_range );

    // m_filter.Random( -1.0f, 1.0f );
    param_range = sqrt( 6.0f / (float)(1.0 + n_order) );
    m_filter.Random( -param_range, param_range );
}    // end of sFSMNLayer


void sFSMNLayer::Prepare( const ExtraInfo& info ) {
    m_block_diagonal.Reshape( info.rank, info.rank, kSetZero );
    m_block_diagonal.SfsmnBlockDiaongal( info.length, m_filter );

    m_length.Reshape( info.length.Rows(), info.length.Columns() );
    m_length.Copy( info.length );
}
         

void sFSMNLayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    m_linear.Compute( feature, output );

    m_memory.Reshape( feature.Rows(), feature.Columns() );
    m_memory.Strmm( 1.0f, 
                    m_block_diagonal, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                     feature, CUBLAS_OP_N );        // @xmb20160226
    // m_memory.Sgemm( 0.0f, 1.0f, m_block_diagonal, CUBLAS_OP_N, feature, CUBLAS_OP_N );

    output->Sgemm( 1.0f, 1.0f, m_memory, CUBLAS_OP_N, m_weight, CUBLAS_OP_N );
    output->ReLU( *output );
}


void sFSMNLayer::BackPropagate( MatrixBase& outDiff, 
                                const MatrixBase& in,
                                const MatrixBase& out,
                                float learningRate,
                                MatrixBase* inDiff ) {
    outDiff.DiffReLU( out, outDiff );
    m_linear.BackPropagate( outDiff, in, Matrix(), learningRate, inDiff );

    if ( m_w_momentum.Rows() != m_weight.Rows() || 
         m_w_momentum.Columns() != m_weight.Columns() ) {
        m_w_momentum.Reshape( m_weight.Rows(), m_weight.Columns(), kSetZero );
    }    

    float avg_lr = -learningRate / (float)outDiff.Rows();

    m_w_momentum.Sgemm( m_momentum,
                        avg_lr,
                        m_memory,
                        CUBLAS_OP_T,
                        outDiff,
                        CUBLAS_OP_N );

    if ( m_weight_decay != 0.0f ) {
        m_w_momentum.Add( 1.0f, -learningRate * m_weight_decay, m_weight );
    }

    m_memory_diff.Reshape( m_memory.Rows(), m_memory.Columns() );
    m_memory_diff.Sgemm( 0.0f, 1.0f, outDiff, CUBLAS_OP_N, m_weight, CUBLAS_OP_T );

    if ( NULL != inDiff ) {
        inDiff->Sgemm( 1.0f, 
                       1.0f, 
                       m_block_diagonal, 
                       CUBLAS_OP_T, 
                       m_memory_diff, 
                       CUBLAS_OP_N );
    }

    m_diagonal_diff.Reshape( m_block_diagonal.Rows(), m_block_diagonal.Columns() );
    m_diagonal_diff.Sgemm( 0.0f,
                           avg_lr / in.Columns(),
                           m_memory_diff,
                           CUBLAS_OP_N,
                           in,
                           CUBLAS_OP_T );

    if ( m_norm_type == "Clip" ) {
        float range = m_norm_param * (-avg_lr);
        m_w_momentum.Clip( -range, range );
        m_diagonal_diff.Clip( -range, range );
    }

    if ( m_norm_type == "L2Norm" ) {
        float w_norm = m_w_momentum.L2Norm() * (-avg_lr);
        if ( w_norm > m_norm_param ) {
            m_w_momentum.Scale( 1.0f / w_norm );
        }
        float d_norm = m_diagonal_diff.L2Norm() * (-avg_lr);
        if ( d_norm > m_norm_param ) {
            m_diagonal_diff.Scale( 1.0f / d_norm );
        }
    }

    m_weight.Add( 1.0f, 1.0f, m_w_momentum );
    m_filter.UpdateSfsmnFilter( m_length, m_diagonal_diff );
}


void sFSMNLayer::SaveParam( ofstream& ofs ) {
    m_linear.SaveParam( ofs );

    vector<float> param;
    m_weight.GetData( &param );
    int size = param.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );

    m_filter.GetData( &param );
    size = param.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );
}


void sFSMNLayer::LoadParam( ifstream& ifs ) {
    m_linear.LoadParam( ifs );

    int size = 0;
    vector<float> param;

    ifs.read( (char*)&size, sizeof(int) );    
    ASSERT( size > 0 );
    param.resize( size );
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_weight.SetData( param );

    ifs.read( (char*)&size, sizeof(int) );    
    ASSERT( size > 0 );
    param.resize( size );
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_filter.SetData( param );
}


void sFSMNLayer::SaveTrainingBuffer( ofstream& ofs ) {
    m_linear.SaveTrainingBuffer( ofs );

    vector<float> buffer;
    m_w_momentum.GetData( &buffer );
    int size = buffer.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&buffer[0], sizeof(float) * size );
}


void sFSMNLayer::LoadTrainingBuffer( ifstream& ifs ) {
    m_linear.LoadTrainingBuffer( ifs );

    int size = 0;
    vector<float> buffer;

    ifs.read( (char*)&size, sizeof(int) );
    ASSERT( size > 0 );
    buffer.resize( size );
    ifs.read( (char*)&buffer[0], sizeof(float) * size );
    m_w_momentum.SetData( buffer );
}


string sFSMNLayer::Report() {
    ostringstream oss;
    oss << "<sFSMN-layer>\n" << m_linear.m_report() << endl;
    oss << "  m_weight': max(" << m_weight.Max()
        << "), min(" << m_weight.Min()
        << ") avg(" << (m_weight.Sum() / m_weight.Rows() / m_weight.Columns()) << ")" << endl;
    SubMatrix __filter( m_filter, MatrixRange( 0, 1, 1, m_filter.Columns() ) );
    oss << "  m_filter: max(" << __filter.Max()
        << "), min(" << __filter.Min()
        << ") avg(" << (__filter.Sum() / __filter.Rows() / __filter.Columns()) << ")";
    return oss.str();
}

// =================================================================================

vFSMNLayer::vFSMNLayer( int n_in, int n_out, int n_order, 
                        float momentum, float weight_decay,
                        string norm_type, float norm_param ):
        m_linear( n_in, n_out, momentum, weight_decay, norm_type, norm_param ),
        m_weight( n_in, n_out ),
        m_filter( n_order, n_in ),
        m_momentum( momentum ),
        m_weight_decay( weight_decay ),
        m_norm_type( norm_type ),
        m_norm_param( norm_param ) {
    float param_range = sqrt( 6.0f / (float)(n_in + n_out) );
    m_weight.Random( -param_range, param_range );

    param_range = sqrt( 6.0f / (float)(1 + n_order) );
    m_filter.Random( -param_range, param_range );
}

void vFSMNLayer::Prepare( const ExtraInfo& info ) {
    m_position.Reshape( info.position.Rows(), info.position.Columns() );
    m_position.Copy( info.position );
}
         

void vFSMNLayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    m_linear.Compute( feature, output );

    m_memory.Reshape( feature.Rows(), feature.Columns() );
    m_memory.VfsmnMemory( feature, m_filter, m_position );

    output->Sgemm( 1.0f, 1.0f, m_memory, CUBLAS_OP_N, m_weight, CUBLAS_OP_N );
    output->ReLU( *output );
}


void vFSMNLayer::BackPropagate( MatrixBase& outDiff, 
                                const MatrixBase& in,
                                const MatrixBase& out,
                                float learningRate,
                                MatrixBase* inDiff ) {
    outDiff.DiffReLU( out, outDiff );
    m_linear.BackPropagate( outDiff, in, Matrix(), learningRate, inDiff );

    if ( m_w_momentum.Rows() != m_weight.Rows() || 
         m_w_momentum.Columns() != m_weight.Columns() ) {
        m_w_momentum.Reshape( m_weight.Rows(), m_weight.Columns(), kSetZero );
    }    

    float avg_lr = -learningRate / (float)outDiff.Rows();

    m_w_momentum.Sgemm( m_momentum,
                        avg_lr,
                        m_memory,
                        CUBLAS_OP_T,
                        outDiff,
                        CUBLAS_OP_N );

    if ( m_weight_decay != 0.0f ) {
        m_w_momentum.Add( 1.0f, -learningRate * m_weight_decay, m_weight );
    }

    m_memory_diff.Reshape( m_memory.Rows(), m_memory.Columns() );
    m_memory_diff.Sgemm( 0.0f, 1.0f, outDiff, CUBLAS_OP_N, m_weight, CUBLAS_OP_T );

    if ( NULL != inDiff ) {
        inDiff->ComputeVfsmnHiddenDiff( m_memory_diff, m_filter, m_position );
    }

    if ( m_norm_type == "Clip" ) {
        float range = m_norm_param * (-avg_lr);
        m_w_momentum.Clip( -range, range );
    }

    if ( m_norm_type == "L2Norm" ) {
        float w_norm = m_w_momentum.L2Norm() * (-avg_lr);
        if ( w_norm > m_norm_param ) {
            m_w_momentum.Scale( 1.0f / w_norm );
        }
    }

    m_filter.UpdateVfsmnFilter( m_memory_diff, in, m_position, avg_lr );
    m_weight.Add( 1.0f, 1.0f, m_w_momentum );
}


void vFSMNLayer::SaveParam( ofstream& ofs ) {
    m_linear.SaveParam( ofs );

    vector<float> param;
    m_weight.GetData( &param );
    int size = param.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );

    m_filter.GetData( &param );
    size = param.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );
}


void vFSMNLayer::LoadParam( ifstream& ifs ) {
    m_linear.LoadParam( ifs );

    int size = 0;
    vector<float> param;

    ifs.read( (char*)&size, sizeof(int) );    
    ASSERT( size > 0 );
    param.resize( size );
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_weight.SetData( param );

    ifs.read( (char*)&size, sizeof(int) );    
    ASSERT( size > 0 );
    param.resize( size );
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_filter.SetData( param );
}


void vFSMNLayer::SaveTrainingBuffer( ofstream& ofs ) {
    m_linear.SaveTrainingBuffer( ofs );

    vector<float> buffer;
    m_w_momentum.GetData( &buffer );
    int size = buffer.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&buffer[0], sizeof(float) * size );
}


void vFSMNLayer::LoadTrainingBuffer( ifstream& ifs ) {
    m_linear.LoadTrainingBuffer( ifs );

    int size = 0;
    vector<float> buffer;

    ifs.read( (char*)&size, sizeof(int) );
    ASSERT( size > 0 );
    buffer.resize( size );
    ifs.read( (char*)&buffer[0], sizeof(float) * size );
    m_w_momentum.SetData( buffer );
}


string vFSMNLayer::Report() {
    ostringstream oss;
    oss << "<vFSMN-layer>\n" << m_linear.m_report() << endl;
    oss << "  m_weight': max(" << m_weight.Max()
        << "), min(" << m_weight.Min()
        << ") avg(" << (m_weight.Sum() / m_weight.Rows() / m_weight.Columns()) 
        << ")" << endl;
    oss << "  m_filter: max(" << m_filter.Max()
        << "), min(" << m_filter.Min()
        << ") avg(" << (m_filter.Sum() / m_filter.Rows() / m_filter.Columns()) 
        << ")";
    return oss.str();
}


// =================================================================================

FofeLayer::FofeLayer( int n_vocab, 
                      int n_projection, 
                      float alpha, 
                      float momentum, 
                      float weight_decay, 
                      string norm_type, 
                      float norm_param )
    : m_projection( n_vocab, n_projection, momentum, weight_decay, norm_type, norm_param ),
      m_alpha( alpha ) {
}    // end of FofeLayer



void FofeLayer::Prepare( const ExtraInfo& info ) {
    m_block_diagonal.Reshape( info.rank, info.rank, kSetZero );
    m_block_diagonal.FOFE( info.length, m_alpha );
}


void FofeLayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    m_buffer.Reshape( output->Rows(), output->Columns(), kUndefined );
    m_projection.Compute( feature, &m_buffer );
    output->Sgemm( 0.0f, 1.0f, m_block_diagonal, CUBLAS_OP_N, m_buffer, CUBLAS_OP_N );
}


void FofeLayer::BackPropagate( MatrixBase& outDiff, 
                               const MatrixBase& in,
                               const MatrixBase& out,
                               float learningRate,
                               MatrixBase* inDiff ) {
    m_buffer.Sgemm( 0.0f, 1.0f, m_block_diagonal, CUBLAS_OP_T, outDiff, CUBLAS_OP_N );
    m_projection.BackPropagate( m_buffer, in, out, learningRate, inDiff );
}


void FofeLayer::SaveParam( ofstream& ofs ) {
    m_projection.SaveParam( ofs );
}


void FofeLayer::LoadParam( ifstream& ifs ) {
    m_projection.LoadParam( ifs );
}


string FofeLayer::Report() {
    return "<FOFE-layer>\n" + m_projection.m_report();
}


// =================================================================================


NCELayer::NCELayer( int n_in, int n_out, float lnZ, string norm_type, float norm_param ) 
    : SoftmaxLayer( n_in, n_out, 0.0f, 0.0f, norm_type, norm_param ),
      m_nce_enabled( false ),
      m_lnZ( lnZ ),
      m_norm_type( norm_type ),
      m_norm_param( norm_param ) {
}


void NCELayer::Prepare( const ExtraInfo& info ) {
    if ( info.nce ) {
        m_nce_enabled = true;

        m_target_and_noise.Reshape( info.targetAndNoise.Rows(), 
                                    info.targetAndNoise.Columns() );
        m_target_and_noise.Copy( info.targetAndNoise );

        m_unigram.Reshape( 1, m_target_and_noise.Rows() );
        m_unigram.GetColumns( info.unigram, m_target_and_noise );
    }
    else {
        m_nce_enabled = false;
    }
    // cout << "checkpoint of prepare" << endl;
}


void NCELayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    if ( m_nce_enabled ) {
        m_partial_weight.Reshape( m_linear.m_weight.Rows(),
                                  m_target_and_noise.Rows() );
        m_partial_bias.Reshape( 1, m_target_and_noise.Rows() );

        m_partial_weight.GetColumns( m_linear.m_weight, m_target_and_noise );
        m_partial_bias.GetColumns( m_linear.m_bias, m_target_and_noise );

        SubMatrix active( *output, 
                          MatrixRange( 0, output->Rows(), 0, m_target_and_noise.Rows() ) );
        active.Sgemm( 0.0f, 1.0f, feature, CUBLAS_OP_N, m_partial_weight, CUBLAS_OP_N );
        // active.Add( 1.0f, 1.0f, m_partial_bias );
        active.Add( 1.0f, m_partial_bias );        // @xmb20160226
        active.Shift( -m_lnZ );
        active.Exp( active );
    }
    else {
        SoftmaxLayer::Compute( feature, output );
        // cout << "checkpoint of compute" << endl;
    }
}


void NCELayer::BackPropagate( MatrixBase& out, 
                              const MatrixBase& in,
                              const MatrixBase& target,
                              float learningRate,
                              MatrixBase* inDiff ) {
    if ( m_nce_enabled ) {
        SubMatrix active( out, 
                          MatrixRange( 0, out.Rows(), 0, m_target_and_noise.Rows() ) );
        active.DiffNCE( active, m_unigram );

        if ( NULL != inDiff ) {
            inDiff->Sgemm( 0.0f, 1.0f, active, CUBLAS_OP_N, m_partial_weight, CUBLAS_OP_T );
        }

        m_partial_w_gradient.Reshape( m_partial_weight.Rows(), m_partial_weight.Columns() );
        m_partial_w_gradient.Sgemm( 0.0f, 1.0f, in, CUBLAS_OP_T, active, CUBLAS_OP_N );

        m_partial_b_gradient.Reshape( m_partial_bias.Rows(), m_partial_bias.Columns() );
        m_partial_b_gradient.SumRowsOf( active, 0.0f, 1.0f );

        if ( m_norm_type == "Clip" ) {
            m_partial_w_gradient.Clip( -m_norm_param, m_norm_param );
            m_partial_b_gradient.Clip( -m_norm_param, m_norm_param );
        }

        if ( m_norm_type == "L2Norm" ) {
            float w_norm = m_partial_w_gradient.L2Norm();
            if ( w_norm > m_norm_param ) {
                m_partial_w_gradient.Scale( 1.0f / w_norm );
            }

            float b_norm = m_partial_b_gradient.L2Norm();
            if ( b_norm > m_norm_param ) {
                m_partial_b_gradient.Scale( 1.0f / b_norm );
            }
        }

        float avg_lr = -learningRate / (float)out.Rows();
        m_linear.m_weight.AddColumns( m_partial_w_gradient, m_target_and_noise, avg_lr );
        m_linear.m_bias.AddColumns( m_partial_b_gradient, m_target_and_noise, avg_lr );
    }
    else  {
        SoftmaxLayer::BackPropagate( out, in, target, learningRate, inDiff );
        // cout << "checkpoint of back-propagate" << endl;
    }
}

// =================================================================================

ProjectionLayer::ProjectionLayer( int n_vocab, int n_projection, 
                                  float momentum, float weight_decay,
                                  string norm_type, float norm_param )
    : m_projection( n_vocab, n_projection ),
      m_momentum( momentum ),
      m_weight_decay( weight_decay ),
      m_norm_type( norm_type ),
      m_norm_param( norm_param ) {
    // float param_range = 1. / sqrt((float)n_vocab);
    float param_range = sqrt( 6.0f / (float)(n_projection + n_vocab) );
    m_projection.Random( -param_range, param_range );
}    // end of ProjectionLayer


void ProjectionLayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    int n_order = feature.Columns();
    int n_example = feature.Rows();
    int n_project = m_projection.Columns();

    assert( feature.Rows() == output->Rows() );
    assert( output->Columns() == n_order * m_projection.Columns() );

    for ( int i = 0; i < n_order; i++ ) {
        SubMatrix gram( *output, 
                        MatrixRange(0, n_example, i * n_project, (i + 1) * n_project ) );
        SubMatrix index( feature,
                         MatrixRange(0, n_example, i, i + 1) );
        gram.GetRows( m_projection, index );
    }
}    // end of ForwardPropagate


void ProjectionLayer::BackPropagate( MatrixBase& outDiff, 
                                     const MatrixBase& in,
                                     const MatrixBase& out,
                                     float learningRate,
                                     MatrixBase* inDiff ) {
    int n_example = outDiff.Rows();
    int n_project = m_projection.Columns();
    int n_order = outDiff.Columns() / n_project;
    ASSERT( n_order == in.Columns() );

    bool useMomentum = m_momentum != 0.0f && m_weight_decay != 0.0f;

    if ( m_norm_type == "Clip" ) {
        outDiff.Clip( -m_norm_param, m_norm_param );
    }
    if ( m_norm_type == "L2Norm" ) {
        float norm = outDiff.L2Norm();
        if ( norm > m_norm_param ) {
            outDiff.Scale( 1.0f / norm );
        }
    }

    if (  useMomentum && 
         (m_gradient.Rows() != m_projection.Rows() || m_gradient.Columns() != m_projection.Columns() ) ) {
        m_gradient.Reshape( m_projection.Rows(), m_projection.Columns(), kSetZero );
    }

    for ( int i = 0; i < n_order; i++ ) {
        SubMatrix gradient( outDiff, 
                            MatrixRange(0, n_example, i * n_project, (i + 1) * n_project ) );
        SubMatrix index( in,
                         MatrixRange(0, n_example, i, i + 1) );

        if ( useMomentum ) {
            m_gradient.Add( m_momentum, -learningRate * m_weight_decay, m_projection );
            m_gradient.AddRows( gradient, index, -learningRate / n_example );
        }
        else {
            m_projection.AddRows( gradient, index, -learningRate / n_example );
        }
    }

    if ( useMomentum ) {
        m_projection.Add( 1.0f, 1.0f, m_gradient );
    }
}    // end of BackPropagate


void ProjectionLayer::SaveParam( ofstream& ofs ) {
    vector<float> param;
    m_projection.GetData( &param );

    int size = param.size();
    ASSERT( size > 0 );

    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );
}    // end of SaveParam


void ProjectionLayer::LoadParam( ifstream& ifs ) {
    int size = 0;
    ifs.read( (char*)&size, sizeof(int) );
    ASSERT( size > 0 );

    vector<float> param( size );
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_projection.SetData( param );
}    // end of LoadParam


string ProjectionLayer::m_report() {
    ostringstream oss;
    oss << "  m_projection: max(" << m_projection.Max() 
        << "), min(" << m_projection.Min() << "), avg(" 
        << (m_projection.Sum() / m_projection.Rows() / m_projection.Columns()) << ")";
    return oss.str();
}


string ProjectionLayer::Report() {
    return "<projection-layer>\n" + m_report(); 
}

// =================================================================================

LinearLayer::LinearLayer( int n_in, int n_out, float momentum, float weight_decay, 
                          string norm_type, float norm_param, bool has_bias ) 
    : m_weight( n_in, n_out ), 
      m_bias( 1, n_out ), 
      m_momentum( momentum ),
      m_weight_decay( weight_decay ),
      m_norm_type( norm_type ),
      m_norm_param( norm_param ),
      m_has_bias( has_bias ) {
    // float param_range = 1.0f / sqrt( (float)n_in );
    float param_range = sqrt( 6.0f / (float)(n_in + n_out) );
    m_weight.Random( -param_range, param_range );
    m_bias.Random( -param_range, param_range );
}    // end of LinearLayer


void LinearLayer::Compute( const MatrixBase& input, MatrixBase* output ) {
    output->Sgemm( 0.0f, 1.0f, input, CUBLAS_OP_N, m_weight, CUBLAS_OP_N );
    if ( m_has_bias ) {
        output->Add( 1.0f, m_bias );    // @xmb20160226
    }
    // output->Add( 1.0f, 1.0f, m_bias );
}    // end of Compute


void LinearLayer::Compute( const SparseMatrix& input, MatrixBase* output ) {
    output->Sgemm( 0.0f, 1.0f, 
                   input, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   m_weight, CUSPARSE_OPERATION_NON_TRANSPOSE );
    if ( m_has_bias ) {
        output->Add( 1.0f, m_bias );
    }
}


void LinearLayer::BackPropagate( MatrixBase& outDiff, 
                                 const MatrixBase& in,
                                 const MatrixBase& out,
                                 float learningRate,
                                 MatrixBase* inDiff ) {
    if ( m_w_momentum.Rows() != m_weight.Rows() || m_w_momentum.Columns() != m_weight.Columns() ) {
        m_w_momentum.Reshape( m_weight.Rows(), m_weight.Columns(), kSetZero );
    }

    if ( m_b_momentum.Rows() != 1 || m_b_momentum.Columns() != m_bias.Columns() ) {
        m_b_momentum.Reshape( 1, m_bias.Columns(), kSetZero );
    }

    if ( NULL != inDiff ) {
        inDiff->Sgemm( 0.0f, 1.0f, outDiff, CUBLAS_OP_N, m_weight, CUBLAS_OP_T );    
    }

    float avg_lr = -learningRate / (float)outDiff.Rows();

    m_w_momentum.Sgemm( m_momentum, 
                        avg_lr,
                        in, 
                        CUBLAS_OP_T,
                        outDiff,
                        CUBLAS_OP_N );

    m_b_momentum.SumRowsOf( outDiff, m_momentum, avg_lr );

    if ( m_weight_decay != 0.0f ) {
        m_w_momentum.Add( 1.0f, -learningRate * m_weight_decay, m_weight );
        m_b_momentum.Add( 1.0f, -learningRate * m_weight_decay, m_bias );        
    }

    if ( m_norm_type == "Clip" ) {
        float range = m_norm_param * (-avg_lr);
        m_w_momentum.Clip( -range, range );
        m_b_momentum.Clip( -range, range );
    }

    if ( m_norm_type == "L2Norm" ) {
        float w_norm = m_w_momentum.L2Norm() * (-avg_lr);
        if ( w_norm > m_norm_param ) {
            m_w_momentum.Scale( 1.0f / w_norm );
        }

        float b_norm = m_b_momentum.L2Norm() * (-avg_lr);
        if ( b_norm > m_norm_param ) {
            m_b_momentum.Scale( 1.0f / b_norm );
        }
    }

    m_weight.Add( 1.0f, 1.0f, m_w_momentum );

    if ( m_has_bias ) {
        m_bias.Add( 1.0f, 1.0f, m_b_momentum );
    }
}    // end of BackPropagate



void LinearLayer::BackPropagate( MatrixBase& outDiff, 
                                 const SparseMatrix& in,
                                 const MatrixBase& out,
                                 float learningRate ) {
    if ( m_w_momentum.Rows() != m_weight.Rows() || m_w_momentum.Columns() != m_weight.Columns() ) {
        m_w_momentum.Reshape( m_weight.Rows(), m_weight.Columns(), kSetZero );
    }

    if ( m_b_momentum.Rows() != 1 || m_b_momentum.Columns() != m_bias.Columns() ) {
        m_b_momentum.Reshape( 1, m_bias.Columns(), kSetZero );
    }

    float avg_lr = -learningRate / (float)outDiff.Rows();

    m_w_momentum.Sgemm( m_momentum, 
                        avg_lr,
                        in, 
                        CUSPARSE_OPERATION_TRANSPOSE,
                        outDiff,
                        CUSPARSE_OPERATION_NON_TRANSPOSE );

    m_b_momentum.SumRowsOf( outDiff, m_momentum, avg_lr );

    if ( m_weight_decay != 0.0f ) {
        m_w_momentum.Add( 1.0f, -learningRate * m_weight_decay, m_weight );
        m_b_momentum.Add( 1.0f, -learningRate * m_weight_decay, m_bias );        
    }

    if ( m_norm_type == "Clip" ) {
        float range = m_norm_param * (-avg_lr);
        m_w_momentum.Clip( -range, range );
        m_b_momentum.Clip( -range, range );
    }

    if ( m_norm_type == "L2Norm" ) {
        float w_norm = m_w_momentum.L2Norm() * (-avg_lr);
        if ( w_norm > m_norm_param ) {
            m_w_momentum.Scale( 1.0f / w_norm );
        }

        float b_norm = m_b_momentum.L2Norm() * (-avg_lr);
        if ( b_norm > m_norm_param ) {
            m_b_momentum.Scale( 1.0f / b_norm );
        }
    }

    m_weight.Add( 1.0f, 1.0f, m_w_momentum );

    if ( m_has_bias ) {
        m_bias.Add( 1.0f, 1.0f, m_b_momentum );
    } 
}    // end of BackPropagate



void LinearLayer::SaveParam( ofstream& ofs ) {
    int size = 0;
    vector<float> param;

    m_weight.GetData( &param );
    size = param.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );

    m_bias.GetData( &param );
    size = param.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&param[0], sizeof(float) * size );
}    // end of SaveParam


void LinearLayer::LoadParam( ifstream& ifs ) {
    int size = 0;
    vector<float> param;

    ifs.read( (char*)&size, sizeof(int) );    
    ASSERT( size > 0 );
    param.resize( size );
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_weight.SetData( param );

    ifs.read( (char*)&size, sizeof(int) );
    ASSERT( size > 0 );
    param.resize( size );    
    ifs.read( (char*)&param[0], sizeof(float) * size );
    m_bias.SetData( param );
}    // end of LoadParam


void LinearLayer::SaveTrainingBuffer( ofstream& ofs ) {
    int size = 0;
    vector<float> buffer;

    m_w_momentum.GetData( &buffer );
    size = buffer.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&buffer[0], sizeof(float) * size );

    m_b_momentum.GetData( &buffer );
    size = buffer.size();
    ASSERT( size > 0 );
    ofs.write( (char*)&size, sizeof(int) );
    ofs.write( (char*)&buffer[0], sizeof(float) * size );
}    // end of SaveTrainingBuffer


void LinearLayer::LoadTrainingBuffer( ifstream& ifs ) {
    int size = 0;
    vector<float> buffer;

    ifs.read( (char*)&size, sizeof(int) );
    ASSERT( size > 0 );
    buffer.resize( size );
    ifs.read( (char*)&buffer[0], sizeof(float) * size );
    m_w_momentum.SetData( buffer );

    ifs.read( (char*)&size, sizeof(int) );
    ASSERT( size > 0 );
    buffer.resize( size );
    ifs.read( (char*)&buffer[0], sizeof(float) * size );
    m_b_momentum.SetData( buffer );
}    // end of LoadTrainingBuffer


string LinearLayer::Report() {
    return "<linear-layer>\n" + m_report();
}


string LinearLayer::m_report() {
    ostringstream oss;
    oss << "  m_weight: max(" << m_weight.Max() << "), min(" << m_weight.Min() << "), avg(" 
        << (m_weight.Sum() / m_weight.Rows() / m_weight.Columns()) << ")" << endl;
    oss << "  m_bias: max(" << m_bias.Max() << "), min(" << m_bias.Min() << "), avg(" 
        << (m_bias.Sum() / m_bias.Rows() / m_bias.Columns()) << ")";
    return oss.str();
}


void LinearLayer::GetWeights( vector<float>* weights ) {
    m_weight.GetData( weights );
}


void LinearLayer::SetWeights( const vector<float>& weights ) {
    m_weight.SetData( weights );
}

// =================================================================================

SoftmaxLayer::SoftmaxLayer( int n_in, int n_out, float momentum, float weight_decay,
                            string norm_type, float norm_param )
    : m_linear( n_in, n_out, momentum, weight_decay, norm_type, norm_param ) {
}    // end of SoftmaxLayer


void SoftmaxLayer::Compute( const MatrixBase& input, MatrixBase* output ) {
    m_linear.Compute( input, output );
    output-> Softmax( *output );
}    // end of Compute


void SoftmaxLayer::BackPropagate( MatrixBase& out, 
                                  const MatrixBase& in,
                                  const MatrixBase& target,
                                  float learningRate,
                                  MatrixBase* inDiff ) {
    out.DiffXent( out, target );
    m_linear.BackPropagate( out, in, Matrix(), learningRate, inDiff );
}    // end of BackPropagate


void SoftmaxLayer::SaveParam( ofstream& ofs ) {
    m_linear.SaveParam( ofs );
}    // end of SaveParam


void SoftmaxLayer::LoadParam( ifstream& ifs ) {
    m_linear.LoadParam( ifs );
}    // end of LoadParam


void SoftmaxLayer::SaveTrainingBuffer( ofstream& ofs ) {
    m_linear.SaveTrainingBuffer( ofs );
}    // end of SaveTrainingBuffer


void SoftmaxLayer::LoadTrainingBuffer( ifstream& ifs ) {
    m_linear.LoadTrainingBuffer( ifs );
}    // end of LoadTrainingBuffer 


string SoftmaxLayer::Report() {
    return "<softmax-layer>\n" + m_linear.m_report();
}

// =================================================================================

ReLULayer::ReLULayer( int n_in, int n_out, float momentum, float weight_decay,
                      string norm_type, float norm_param )
    : m_linear( n_in, n_out, momentum, weight_decay, norm_type, norm_param ) {
}    // end of ReLULayer


void ReLULayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    m_linear.Compute( feature, output );
    output->ReLU( *output );
}    // end of Compute


void ReLULayer::BackPropagate( MatrixBase& outDiff, 
                               const MatrixBase& in,
                               const MatrixBase& out,
                               float learningRate,
                               MatrixBase* inDiff ) {
    outDiff.DiffReLU( out, outDiff );
    m_linear.BackPropagate( outDiff, in, Matrix(), learningRate, inDiff );
}    // end of BackPropagate


void ReLULayer::SaveParam( ofstream& ofs ) {
    m_linear.SaveParam( ofs );
}    // end of SaveParam


void ReLULayer::LoadParam( ifstream& ifs ) {
    m_linear.LoadParam( ifs );
}    // end of LoadParam


void ReLULayer::SaveTrainingBuffer( ofstream& ofs ) {
    m_linear.SaveTrainingBuffer( ofs );
}    // end of SaveTrainingBuffer


void ReLULayer::LoadTrainingBuffer( ifstream& ifs ) {
    m_linear.LoadTrainingBuffer( ifs );
}    // end of LoadTrainingBuffer 


string ReLULayer::Report() {
    return "<ReLU-layer>\n" + m_linear.m_report();
}

// =================================================================================

SigmoidLayer::SigmoidLayer( int n_in, int n_out, float momentum, float weight_decay,
                            string norm_type, float norm_param )
    : m_linear( n_in, n_out, momentum, weight_decay, norm_type, norm_param ) {
}    // end of ReLULayer


void SigmoidLayer::Compute( const MatrixBase& feature, MatrixBase* output ) {
    m_linear.Compute( feature, output );
    output->Sigmoid( *output );
}    // end of Compute


void SigmoidLayer::BackPropagate( MatrixBase& outDiff, 
                                     const MatrixBase& in,
                                  const MatrixBase& out,
                                  float learningRate,
                                  MatrixBase* inDiff ) {
    outDiff.DiffSigmoid( out, outDiff );
    m_linear.BackPropagate( outDiff, in, Matrix(), learningRate, inDiff );
}    // end of BackPropagate


void SigmoidLayer::SaveParam( ofstream& ofs ) {
    m_linear.SaveParam( ofs );
}    // end of SaveParam


void SigmoidLayer::LoadParam( ifstream& ifs ) {
    m_linear.LoadParam( ifs );
}    // end of LoadParam


void SigmoidLayer::SaveTrainingBuffer( ofstream& ofs ) {
    m_linear.SaveTrainingBuffer( ofs );
}    // end of SaveTrainingBuffer


void SigmoidLayer::LoadTrainingBuffer( ifstream& ifs ) {
    m_linear.LoadTrainingBuffer( ifs );
}    // end of LoadTrainingBuffer 


string SigmoidLayer::Report() {
    return "<sigmoid-layer>\n" + m_linear.m_report();
}

// =================================================================================

#ifdef LAYER_UNIT_TEST
#include <fstream>
using namespace std;

const int BYTES_OF_INT = 4;
const int BITS_OF_CHAR = 8;
const int NUM_OF_DIGITS = 10;


inline int readInt( ifstream& infile ) {
    unsigned char c = 0;
    int to_return = 0;
    for ( int i = 0; i < BYTES_OF_INT; i++ ) {
        c = infile.get();
        to_return = (to_return << BITS_OF_CHAR) | c;
    }
    return to_return;
}    // end of readInt


void getImage( const char* filename, vector<float>& image ) {
    ifstream infile;
    infile.open( filename, ios::binary | ios::in );
    if ( !infile.good() ) {
        fprintf( stderr, "%10s %4d %s failed\n", __FILE__, __LINE__, "ifstream::open()" );
        exit( EXIT_FAILURE );
    }

    int magic = readInt( infile );
    int nExample = readInt( infile );
    int nRows = readInt( infile );
    int nColumns = readInt( infile );
    int nDimension = nRows * nColumns;

    image.clear();
    image.resize( nExample * nDimension );

    for ( int i = 0; i < nExample; i++ ) {
        for ( int j = 0; j < nDimension; j++ ) {
            image[i * nDimension + j] = infile.get() / 128.0f - 1.0f;
        }
    }

    infile.close();
}    // end of getImage


void getLabel( const char* filename, vector<float>& label ) {
    ifstream infile;
    infile.open( filename, ios::binary | ios::in );
    if ( !infile.good() ) {
        fprintf( stderr, "%10s %4d %s failed\n", __FILE__, __LINE__, "ifstream::open()" );
        exit( EXIT_FAILURE );
    }

    int magic = readInt( infile );
    int nLabels = readInt( infile );

    label.resize( nLabels );

    for ( int i = 0; i < nLabels; i++ ) {
        label[i] = (float)infile.get();
    }
}    // end of getLabel


float getErrorRate( const vector<float>& predicted, const vector<float>& expected ) {
    int n_example = predicted.size();
    assert ( n_example == expected.size() );

    int err = 0;
    for ( int i = 0; i < n_example; i++ ) {
        if ( predicted[i] != expected[i] ) {
            err++;
        }
    }

    return 100.0f * err / n_example;
}    // end of getErrorRate


int main( int argc, char** argv ) {
    vector<float> trainImage;
    vector<float> trainLabel;

    vector<float> testImage;
    vector<float> testLabel;

    getImage( "/eecs/research/asr/mingbin/train-images.idx3-ubyte", trainImage );
    getLabel( "/eecs/research/asr/mingbin/train-labels.idx1-ubyte", trainLabel );
    cout << "Training set loaded" << endl;

    getImage( "/eecs/research/asr/mingbin/t10k-images.idx3-ubyte", testImage );
    getLabel( "/eecs/research/asr/mingbin/t10k-labels.idx1-ubyte", testLabel );
    cout << "Test set loaded" << endl;

    int nDimension = trainImage.size() / trainLabel.size();
    assert( nDimension == testImage.size() / testLabel.size() );

    Matrix train_feature( trainLabel.size(), nDimension );
    Matrix train_target( trainLabel.size(), 1 );
    train_feature.SetData( trainImage );
    train_target.SetData( trainLabel );

    Matrix test_feature( testLabel.size(), nDimension );
    Matrix test_target( testLabel.size(), 1 );
    test_feature.SetData( testImage );
    test_target.SetData( testLabel );

    float every1Kcost = 0.0f;

    SoftmaxLayer trainer( nDimension, NUM_OF_DIGITS );
    const float LR = 0.003f;
    const int BATCH_SIZE = 1;

    Matrix output( BATCH_SIZE, NUM_OF_DIGITS );

    for ( int i = 0; i < train_target.Rows(); i += BATCH_SIZE ) {
        SubMatrix input( train_feature, 
                         MatrixRange(i, i + BATCH_SIZE, 0, train_feature.Columns()) );
        SubMatrix target( train_target, MatrixRange(i, i + BATCH_SIZE, 0, 1) );
        trainer.Compute( input, &output );
        every1Kcost += output.Xent( target );
        trainer.BackPropagate( output, input, target, LR, NULL );

        if ( (i + BATCH_SIZE) % 1000 == 0 ) {
            cout << setw(6) << (i + BATCH_SIZE) << " : " << every1Kcost << endl; 
            every1Kcost = 0.0f;
        }
    }

    output.Reshape( testLabel.size(), NUM_OF_DIGITS );
    trainer.Compute( test_feature, &output );

    Matrix predicted( 1, testLabel.size() );
    predicted.ArgMax( output );

    vector<float> result;
    predicted.GetData( &result );
    float errRate = getErrorRate( result, testLabel );
    cout << "error rate: " << errRate << '%' << endl;

    return EXIT_SUCCESS;
}

#endif
