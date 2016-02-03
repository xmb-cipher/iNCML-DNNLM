/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : network.cpp
Last Update : Feb 2, 2016
Description : Provide concrete implementation of neural network
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -o main -lcublas -lcurand -Xcompiler -rdynamic -DNETWORK_UNIT_TEST matrix.cu layer.cpp network.cpp

#include "network.h"

void readConfig( istringstream& iss,
				 int& n_in,
				 int& n_out,
				 int& n_vocab,
				 int& n_project,
				 int& n_order,
				 float& momentum,
				 float& weight_decay,
				 float& alpha,
				 float& lnZ ) {
	n_in = 0;
	n_out = 0;
	n_vocab = 0;
	n_project = 0;
	n_order = 0;
	momentum = 0.0f;
	weight_decay = 0.0f;
	alpha = 0.0f;
	lnZ = 0.0f;

	string token;
	while ( !iss.eof() ) {
		iss >> token;
		if ( token == "vocabulary" ) {
			iss >> n_vocab;
		}
		else if ( token == "projection" ) {
			iss >> n_project;
		}
		else if ( token == "input" ) {
			iss >> n_in;
		}
		else if ( token == "output" ) {
			iss >> n_out;
		}
		else if ( token == "momentum" ) {
			iss >> momentum;
		}
		else if ( token == "weight-decay" ) {
			iss >> weight_decay;
		}
		else if ( token == "forgetting-factor" ) {
			iss >> alpha;
		}
		else if ( token == "order" ) {
			iss >> n_order;
		}
		else if ( token == "lnZ" ) {
			iss >> lnZ;
		}
	}
}	// end of readConfig


Network::Network( const char* config ) {
	ifstream file;
	file.open( config );

	ASSERT( file.is_open() );
	string line;

	while ( getline( file, line ) ) {
		line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
        istringstream iss( line );

        string layer_type;
    	iss >> layer_type;

    	int n_in = 0;
    	int n_out = 0;
    	int n_vocab = 0;
    	int n_project = 0;
    	int n_order = 0;
    	float momentum = 0.0f;
    	float weight_decay = 0.0f;
    	float alpha = 0.7f;
    	float lnZ = 9.0f;

    	if ( layer_type == "<projection-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order, 
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new ProjectionLayer( n_vocab, n_project, momentum, weight_decay ) );
    		cout << left 
    			 << setw(18) << "<projection-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out 
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay << endl;
    	}
    	else if ( layer_type == "<FOFE-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order, 
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new FofeLayer( n_vocab, n_project, alpha, momentum, weight_decay ) );
    		cout << left 
    			 << setw(18) << "<FOFE-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out 
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay 
    			 << " forgetting-factor " << setw(8) << alpha << endl;
    	}
    	else if ( layer_type == "<linear-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order,
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new LinearLayer( n_in, n_out, momentum, weight_decay ) );
    		cout << left
    			 << setw(18) << "<linear-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay << endl; 
    	}
    	else if ( layer_type == "<ReLU-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order, 
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new ReLULayer( n_in, n_out, momentum, weight_decay ) );
    		cout << left
    		     << setw(18) << "<ReLU-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out 
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay << endl; 
    	}
    	else if ( layer_type == "<sigmoid-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order,
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new SigmoidLayer( n_in, n_out, momentum, weight_decay ) );
    		cout << left
    		     << setw(18) << "<sigmoid-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay << endl; 
    	}
    	else if ( layer_type == "<softmax-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order,
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new SoftmaxLayer( n_in, n_out, momentum, weight_decay ) );
    		cout << left
    		     << setw(18) << "<softmax-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay << endl;  
    	}
    	else if ( layer_type == "<sFSMN-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order,
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new sFSMNLayer( n_in, n_out, n_order, momentum, weight_decay ) );
    		cout << left
    		     << setw(18) << "<sFSMN-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay 
    			 << " order " << setw(8) << n_order << endl;  
    	}
    	else if ( layer_type == "<vFSMN-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order,
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new vFSMNLayer( n_in, n_out, n_order, momentum, weight_decay ) );
    		cout << left
    		     << setw(18) << "<vFSMN-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out
    			 << " momentum " << setw(8) << momentum
    			 << " weight-decay " << setw(8) << weight_decay 
    			 << " order " << setw(8) << n_order << endl;  
    	}
    	else if ( layer_type == "<NCE-layer>" ) {
    		readConfig( iss, n_in, n_out, n_vocab, n_project, n_order,
    					momentum, weight_decay, alpha, lnZ );
    		m_layer.push_back( new NCELayer( n_in, n_out, lnZ ) );
    		cout << left
    		     << setw(18) << "<NCE-layer>" << " : "
    			 << setw(10) << n_in << "-->  "
    			 << setw(10) << n_out
    			 << " lnZ " << setw(8) << lnZ << endl;
    	}

    	if ( n_in != 0 && n_out != 0 ) {
    		m_input_size.push_back( n_in );
    		m_output_size.push_back( n_out );
    	}
	}	// end of while ( getline

	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_output.push_back( new Matrix() );
		if ( i == 0 ) {
			m_input.push_back( NULL );
		}
		else {
			m_input.push_back( m_output[i - 1] );
		}
	}

	m_gradient.resize( m_layer.size() - 1 );
	for ( int i = 0; i < m_gradient.size(); i++ ) {
		m_gradient[i] = new Matrix();
	}

	// so far, we would like to assume that the output layer must be softmax-layer
	for ( int i = 0; i < m_layer.size() - 1; i++ ) {
		ASSERT( NULL == dynamic_cast<SoftmaxLayer*>( m_layer[i] ) );
	}
	// ASSERT( NULL != dynamic_cast<SoftmaxLayer*>(m_layer.back()) );
}	// end of Network



Network::~Network() {
	for ( int i = 0; i < m_layer.size(); i++ ) {
		if ( NULL != m_layer[i] ) {
			delete m_layer[i];
			m_layer[i] = NULL;
		}
	}

	for ( int i = 0; i < m_output.size(); i++ ) {
		if ( NULL != m_output[i] ) {
			delete m_output[i];
			m_output[i] = NULL;
		}
	}

	for ( int i = 0; i < m_gradient.size(); i++ ) {
		if ( NULL != m_gradient[i] ) {
			delete m_gradient[i];
			m_gradient[i] = NULL;
		}
	}
}	// end of ~Network



const MatrixBase& Network::Compute( const MatrixBase& feature ) {
	ASSERT( feature.Columns() == m_input_size.front() );
	int n_example = feature.Rows();

	m_input[0] = &feature;
	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_output[i]->Reshape( n_example, m_output_size[i], kSetZero );
	}

	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_layer[i]->Compute( *m_input[i], m_output[i] );
	}

	return *(m_output.back());
}	// end of Compute



void Network::BackPropagate( const MatrixBase& target, float lr ) {
	int n_example = target.Rows();
	ASSERT( n_example == m_output.back()->Rows() );

	for ( int i = 0; i < m_gradient.size(); i++ ) {
		m_gradient[i]->Reshape( n_example, m_output_size[i] );
	}

	for ( int i = m_layer.size() - 1; i >= 0; i-- ) {
		if ( i == m_layer.size() - 1 ) {
			m_layer[i]->BackPropagate( *(m_output.back()),
								       *m_input[i],
								       target,
								       lr,
								       (i == 0 ? NULL : m_gradient[i - 1]) );
		}
		else {
			m_layer[i]->BackPropagate( *m_gradient[i],
									   *m_input[i],
									   *m_output[i],
									   lr,
									   (i == 0 ? NULL : m_gradient[i - 1]) );
		}
	}
}	// end of BackPropagate


void Network::SaveParam( const char* filename ) {
	ofstream ofs( filename, ios::binary );
	ASSERT( ofs.is_open() );
	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_layer[i]->SaveParam( ofs );
	}
	ofs.close();
}	// end of SaveParam


void Network::LoadParam( const char* filename ) {
	ifstream ifs( filename, ios::binary );
	ASSERT( ifs.is_open() );
	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_layer[i]->LoadParam( ifs );
	}
	ifs.close();
}	// end of LoadParam


void Network::SaveTrainingBuffer( const char* filename ) {
	ofstream ofs( filename, ios::binary );
	ASSERT( ofs.is_open() );
	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_layer[i]->SaveParam( ofs );
	}
	ofs.close();
}	// end of SaveTrainingBuffer


void Network::LoadTrainingBuffer( const char* filename ) {
	ifstream ifs( filename, ios::binary );
	ASSERT( ifs.is_open() );
	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_layer[i]->LoadTrainingBuffer( ifs );
	}
	ifs.close();
}	// end of LoadTrainingBuffer


void Network::Prepare( const ExtraInfo& info ) {
	for ( int i = 0; i < m_layer.size(); i++ ) {
		m_layer[i]->Prepare( info );
	}
}	// end of Preapre


void Network::Report() {
	for ( int i = 0; i < m_layer.size(); i++ ) {
		cout << m_layer[i]->Report() << endl;
	}
}

// ==========================================================================================


#ifdef NETWORK_UNIT_TEST

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
}	// end of readInt


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
}	// end of getImage


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
}	// end of getLabel


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
}	// end of getErrorRate

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

	Network trainer( "mnist.config" );
	const float LR = 0.003f;
	const int BATCH_SIZE = 50;

	for ( int i = 0; i < train_target.Rows(); i += BATCH_SIZE ) {
		SubMatrix input( train_feature, 
						 MatrixRange(i, i + BATCH_SIZE, 0, train_feature.Columns()) );
		SubMatrix target( train_target, MatrixRange(i, i + BATCH_SIZE, 0, 1) );
		const MatrixBase& output = trainer.Compute( input );
		every1Kcost += output.Xent( target );
		trainer.BackPropagate( target, LR );
		
		if ( (i + BATCH_SIZE) % 1000 == 0 ) {
			cout << setw(6) << (i + BATCH_SIZE) << " : " << every1Kcost << endl; 
			every1Kcost = 0.0f;
		}
	}

	const MatrixBase& output = trainer.Compute( test_feature );
	Matrix predicted( 1, testLabel.size() );
	predicted.ArgMax( output );

	vector<float> result;
	predicted.GetData( &result );
	float errRate = getErrorRate( result, testLabel );
	cout << "error rate: " << errRate << '%' << endl;


	trainer.SaveParam( "network-unit-test" );
	Network cpy( "mnist.config" );
	cpy.LoadParam( "network-unit-test" );
	const MatrixBase& cpyout = cpy.Compute( test_feature );
	predicted.ArgMax( cpyout );
	predicted.GetData( &result );
	errRate = getErrorRate( result, testLabel );
	cout << "error rate: " << errRate << '%' << endl;

	return EXIT_SUCCESS;
}

#endif
