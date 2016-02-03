/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : batch-constructor.cpp
Last Update : Feb 2, 2016
Description : Provide concrete implemenetation of constructing mini-batch 
			  for SGD in language modelling
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -lcurand -lcublas -Xcompiler -rdynamic -o ../batch-constructor batch-constructor.cpp matrix.cu -DBATCH_CONSTRUCTOR_UNIT_TEST

#include "batch-constructor.h"

BatchConstructor::BatchConstructor( const char* filename, 
									int order, 
									int batchSize, 
									int sampleSize, 
									bool isTrainMode ) :
		m_gpu_buffer( batchSize + sampleSize, order + 3 ),
		m_cpu_buffer( (batchSize + sampleSize) * (order + 3) ),
		m_n_sample( sampleSize ),
		m_batch_size( batchSize ),
		m_n_order( order ),
		m_n_sentence_discarded( 0 ),
		m_n_example_discarded( 0 ),
		m_is_training( isTrainMode ) {
	int length;
	float stop_idx;

	ifstream ifs( filename );
	ASSERT( ifs.is_open() );

	ifs.read( (char*)&m_start_idx, sizeof(float) );
	ifs.read( (char*)&stop_idx, sizeof(float) );

	while ( true ) {
		ifs.read( (char*)&length, sizeof(int) );
		if ( ifs.eof() ) {
			break;
		}

		vector<float> s( 1 + length );
		s[length] = stop_idx;
		ifs.read( (char*)&s[0], sizeof(float) * length );

		m_sentence.push_back( s );/*
	#ifdef BATCH_CONSTRUCTOR_UNIT_TEST
		for ( int i = 0; i < s.size(); i++ ) {
			cout << (int)s[i] << "  ";
		}
		cout << endl;
	#endif*/
	}

	ifs.close();

#ifndef BATCH_CONSTRUCTOR_UNIT_TEST	
	random_shuffle( m_sentence.begin(), m_sentence.end() );
#endif
	m_itr = m_sentence.begin();

	cout << CurrentTime() << ") batch-constructor created: " 
		 << m_sentence.size() << " sentences, "
		 << m_batch_size << " examples/batch, "
		 << m_n_sample << " noise/batch" << endl;
}	// end of BatchConstructor


BatchConstructor::~BatchConstructor() {
}	// end of ~BatchConstructor



void BatchConstructor::Reset() {
	random_shuffle( m_sentence.begin(), m_sentence.end() );
	m_itr = m_sentence.begin();
}	// end of Reset



bool BatchConstructor::HasNext() const {
	return m_itr != m_sentence.end();
}	// end of HasNext



void BatchConstructor::PrepareNext( const MatrixBase* accUnigram, 
									const float* wordCount ) {
	int n_non_truncated = 0;
	int n_sentence = 1;
	int n_example = 0;
	m_cpu_buffer[ m_n_order + 1 ] = 0.0f;

	while ( m_itr != m_sentence.end() ) {
		if ( m_itr->size() > m_batch_size ) {
			m_n_sentence_discarded++;
			m_n_example_discarded += m_itr->size();
			cout << CurrentTime() << ")  "
				 << m_n_example_discarded
				 << " examples ("
				 << m_n_sentence_discarded
				 << " sentence(s)) have been discarded. " << endl;
			m_itr++;
			continue;
		}

		// to decide if the length of the next sentence exceeds a mini-batch
		int n_to_cpy = (n_non_truncated + m_itr->size()) > m_batch_size ?
							(m_batch_size - n_non_truncated) : m_itr->size();

		// the stride of m_cpu_buffer is order + 2
		int stride = m_n_order + 3;
		for ( int i = 0; i < m_n_order + 1; i++ ) {
			int n_start = m_n_order - i;
			n_start = n_start > n_to_cpy ? n_to_cpy : n_start;
			for ( int j = 0; j < n_start; j++ ) {
				m_cpu_buffer[ (n_non_truncated + j) * stride + i ] = m_start_idx;
			}
			for ( int j = n_start; j < n_to_cpy; j++ ){
				m_cpu_buffer[ (n_non_truncated + j) * stride + i ] = (*m_itr)[j - n_start];
			}
		}

		for ( int j = 0; j < n_to_cpy; j++ ) {
			m_cpu_buffer[ (n_non_truncated + j + 1) * stride - 1 ] = (float)j;
		}

		m_cpu_buffer[ n_sentence * stride + m_n_order + 1 ] =  n_non_truncated + n_to_cpy;
		n_sentence++;

		if ( n_non_truncated + m_itr->size() < m_batch_size ) {
			n_non_truncated += m_itr->size();
			m_itr++;
		}
		else {
			if ( n_non_truncated + m_itr->size() == m_batch_size ) {
				n_non_truncated += m_itr->size();
				m_itr++;
			}
			n_example = m_batch_size;
			goto exit;
		}
	}

	n_example = n_non_truncated;

exit:
	m_gpu_buffer.SetData( m_cpu_buffer );

	if ( m_is_training ) {
		m_input = SubMatrix( m_gpu_buffer, MatrixRange( 0, n_example, 0, m_n_order ) );
		m_target = SubMatrix( m_gpu_buffer, MatrixRange( 0, n_example, m_n_order, m_n_order + 1) );
		m_noise = SubMatrix( m_gpu_buffer, 
							 MatrixRange( n_example, n_example + m_n_sample, 
							 			  m_n_order, m_n_order + 1 ) );
		m_target_and_noise = SubMatrix( m_gpu_buffer, 
							 			MatrixRange( 0, n_example + m_n_sample, 
							 						 m_n_order, m_n_order + 1 ) );
		m_position = SubMatrix( m_gpu_buffer,
								MatrixRange( 0, n_example, 
											 m_n_order + 2, m_n_order + 3 ) );
	}
	else {
		if ( n_example > n_non_truncated ) {
			n_sentence--;
		}
		m_input = SubMatrix( m_gpu_buffer, MatrixRange( 0, n_non_truncated, 0, m_n_order ) );
		m_target = SubMatrix( m_gpu_buffer, MatrixRange( 0, n_non_truncated, m_n_order, m_n_order + 1) );
		m_noise = SubMatrix( m_gpu_buffer, 
							 MatrixRange( n_non_truncated, n_non_truncated + m_n_sample, 
							 			  m_n_order, m_n_order + 1 ) );
		m_target_and_noise = SubMatrix( m_gpu_buffer, 
							 			MatrixRange( 0, n_non_truncated + m_n_sample, 
							 						 m_n_order, m_n_order + 1 ) );
		m_position = SubMatrix( m_gpu_buffer,
								MatrixRange( 0, n_non_truncated, 
											 m_n_order + 2, m_n_order + 3 ) );
	}
	m_length = SubMatrix( m_gpu_buffer, 
						  MatrixRange( 0, n_sentence, m_n_order + 1, m_n_order + 2 ) );

	if ( NULL != accUnigram && NULL != wordCount ) {
		m_noise.Random( 0.0f, *wordCount );
		m_noise.LowerBound( m_noise, *accUnigram );
	}
}	// end of PrepareNext



SubMatrix BatchConstructor::GetInput() {
	return m_input;
}	// end of GetInput


SubMatrix BatchConstructor::GetTarget() {
	return m_target;
}


SubMatrix BatchConstructor::GetNoise() {
	return m_noise;
}


SubMatrix BatchConstructor::GetTargetAndNoise() {
	return m_target_and_noise;
}


SubMatrix BatchConstructor::GetSentenceLength() {
	return m_length;
}


SubMatrix BatchConstructor::GetPosition() {
	return m_position;
}



// ==========================================================================================


#ifdef BATCH_CONSTRUCTOR_UNIT_TEST

void GetProbability( ifstream& ifs, Matrix* accProb, Matrix* unigram ) {
	string line;
	string token;
	float p;
	vector<float> buffer;

	while ( getline( ifs, line ) ) {
		line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
		istringstream iss( line );
		iss >> token;
		iss >> token;
		iss >> p;
		buffer.push_back( p );
	}

	for ( int i = 0; i < buffer.size(); i++ ) {
		buffer[i] /= buffer.back();
	}

	accProb->Reshape( 1, buffer.size() );
	accProb->SetData( buffer );

	for ( int i = buffer.size() - 1; i > 0; i-- ) {
		buffer[i] -= buffer[i - 1];
	}

	unigram->Reshape( 1, buffer.size() );
	unigram->SetData( buffer );
}	// end of GetProbability



int main( int argc, char** argv ) {
	float n_noise = 128.0f;
	BatchConstructor bc( "ptb.test.numeric", 2, 256, (int)n_noise, false );
	Matrix hopeless;

	ifstream vocab( "ptb.vocab" );
	ASSERT( vocab.is_open() );
	Matrix accumulativeProbability;
	Matrix unigram;
	GetProbability( vocab, &accumulativeProbability, &unigram );
	vocab.close();

	cout << accumulativeProbability.Rows() << " " 
		 << accumulativeProbability.Columns() << endl;

	cout << unigram.Rows() << " " 
		 << unigram.Columns() << endl;

	accumulativeProbability.Scale( n_noise );
	unigram.Scale( n_noise );

	cout << "data loaded" << endl;

	while ( bc.HasNext() ) {
		bc.PrepareNext( &accumulativeProbability, &n_noise );

		SubMatrix feature = bc.GetInput();
		cout << "  feature" << endl;
		cout << feature << endl;

		SubMatrix target = bc.GetTarget();
		cout << "  target" << endl;
		cout << target << endl;

		SubMatrix noise = bc.GetNoise();
		cout << "  noise" << endl;
		cout << noise << endl;

		SubMatrix length = bc.GetSentenceLength();
		cout << feature.Rows() << endl;
		cout << target.Rows() << endl;
		cout << noise.Rows() << endl;
		cout << "  length" << endl;
		cout << length << endl;

		SubMatrix position = bc.GetPosition();
		cout << "  position" << endl;
		cout << position << endl;
/*
		hopeless.Reshape( feature.Rows(), feature.Rows(), kSetZero );
		hopeless.FOFE( length, 0.7f );
		cout << hopeless << endl;*/
	}

	return EXIT_SUCCESS;
}

#endif