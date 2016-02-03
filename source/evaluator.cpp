/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : evaluator.cpp
Last Update : Feb 2, 2016
Description : Compute PPL of a test set with a trained model
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -O3 -arch=sm_30 -lcurand -lcublas -Xcompiler -rdynamic -o ../evaluator evaluator.cpp network.cpp layer.cpp batch-constructor.cpp matrix.cu

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#include "matrix.h"
#include "network.h"
#include "batch-constructor.h"
#include <cstdlib>
using namespace std;


int main( int argc, char** argv ) {

	if ( argc == 6 || argc == 7 ) {
		const char* config = argv[1];
		const char* model = argv[2];
		const char* data = argv[3];
		int order = atoi( argv[4] );
		int batch = atoi( argv[5] );

		float lnZ = 0.0f;
		if ( argc > 6 ) {
			lnZ = atof( argv[6] );
			ASSERT( lnZ > 0.0f );
		}

		BatchConstructor bc( data, order, batch, 0, false );
		Network network( config );
		network.LoadParam( model );

		Matrix placeholder;
		Matrix buffer;
		double loss = 0.0;
		int nExample = 0;

		while( bc.HasNext() ) {
			bc.PrepareNext();

			const SubMatrix input = bc.GetInput();
			const SubMatrix target = bc.GetTarget();

			ExtraInfo info( input.Rows(),
							bc.GetSentenceLength(),
							false,
							target,			// actually not used
							placeholder );	// not used either
			network.Prepare( info );

			const MatrixBase& output = network.Compute( input );

			if ( lnZ == 0.0f ) {
				loss += output.Xent( target );
				nExample += output.Rows();
			}
			else {
				buffer.Reshape( output.Rows(), output.Columns() );
				buffer.Copy( output );
				buffer.Shift( -lnZ );
				buffer.Exp( buffer );
				loss += buffer.Xent( target );
				nExample += output.Rows();
			}
		}

		double avgLoss = loss / (double)nExample;
		double ppl = exp(avgLoss);
		cout << right
			 << CurrentTime() << ") average cross-entropy loss of " << nExample
			 << " examples: " << KGRN << avgLoss << KNRM << " or "
             << KGRN << ppl << KNRM << " in PPL " << endl;

		return EXIT_SUCCESS;
	}

	else {
		cerr << KRED;
		cerr << "Usag: " << argv[0] << " <config> <model> <numeric-data> <order> <batch-size> [lnZ]" << endl;
		cerr << "    <config>       : network configuration" << endl;
		cerr << "    <model>        : trained model corresponding to config" << endl;
		cerr << "    <numeric-data> : data set to be evaluated, in numeric form" << endl;
		cerr << "    <order>        : length of context window" << endl;
		cerr << "    <batch-size>   : mini-batch size" << endl;
		cerr << "    [lnZ]          : optional, if set, softmax is not applied" << endl;
		cerr << KNRM;
		exit( EXIT_FAILURE );
	}
}