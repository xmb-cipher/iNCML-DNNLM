/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : trainer.cpp
Last Update : Feb 2, 2016
Description : The entry point of LM training
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -O3 -arch=sm_30 -lcurand -lcublas -Xcompiler -rdynamic -o ../trainer trainer.cpp network.cpp layer.cpp batch-constructor.cpp matrix.cu


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
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
using namespace std;


bool fileExists( const char* filename ) {
    ifstream ifs( filename );
    bool result = ifs.is_open();
    if ( result ) {
        ifs.close();
    }
    return result;
}


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
}    // end of GetProbability


int main( int argc, char** argv ) {
    const int minArgc = 6;
    int maxIter = 16;
    float prevPPL = 6479745038;
    bool lrDecayStarted = false;

    if ( argc == minArgc || argc == minArgc + 1 ) {
        string basename = argv[2];

        float lr = atof( argv[3] );
        ASSERT( lr > 0.0f );

        int batchSize = atoi( argv[4] );
        ASSERT( batchSize > 2 );

        int order = atoi( argv[5] );
        ASSERT( order > 0 );
        
        float n_noise = 0.0f;
        if ( argc == minArgc + 1 ) {
            n_noise = atof( argv[6] );
        }

        ifstream vocab( (basename + ".vocab").c_str() );
        ASSERT( vocab.is_open() );
        Matrix accumulativeProbability;
        Matrix unigram;
        GetProbability( vocab, &accumulativeProbability, &unigram );
        vocab.close();
        accumulativeProbability.Scale( n_noise );
        unigram.Scale( n_noise );

        string config = string("config/") + string(argv[1]);
        Network trainer( config.c_str() );
        cout << CurrentTime() << ") neural network created" << endl;
        trainer.Report();

        BatchConstructor* train = NULL;
        BatchConstructor* valid = NULL;
        BatchConstructor* test = NULL;

        string valid_str = "numeric-data/" + basename + ".valid.numeric";
        if ( fileExists( valid_str.c_str() ) ) {
            valid = new BatchConstructor( valid_str.c_str(), order, batchSize * 5, 0, false );
            cout << CurrentTime() << ") validation set loaded" << endl;
        }

        string test_str = "numeric-data/" + basename + ".test.numeric";
        if ( fileExists( test_str.c_str() ) ) {
            test = new BatchConstructor( test_str.c_str(), order, batchSize * 5, 0, false );
            cout << CurrentTime() << ") test set loaded" << endl;
        }

        train = new BatchConstructor( ("numeric-data/" + basename + ".train.numeric").c_str(), 
                                      order, batchSize, (int)n_noise, true );
        cout << CurrentTime() << ") training set loaded" << endl;

        for ( int epoch = 0; epoch < maxIter; epoch++ ) {
            cout << right << CurrentTime()
                 << ") epoch" << setw(3) << epoch 
                 << " starts with learning-rate " << lr << endl;

            ostringstream oss;
            oss << "model-archive/" << argv[1] << ".epoch" << epoch;
            trainer.SaveParam( oss.str().c_str() );
            trainer.SaveTrainingBuffer( (oss.str() + ".buffer").c_str() );

            float loss = 0.0f;
            int exampleCnt = 0;
            long lastReport = time( NULL );

            train->Reset();
            while ( train->HasNext() ) {
                if ( n_noise > 0.0f ) {
                    train->PrepareNext( &accumulativeProbability, &n_noise );
                }
                else {
                    train->PrepareNext();
                }

                SubMatrix input = train->GetInput();
                SubMatrix target = train->GetTarget();

                ExtraInfo info( input.Rows(), 
                                train->GetSentenceLength(),
                                n_noise > 0.0f,
                                train->GetTargetAndNoise(),
                                unigram,
                                train->GetPosition() );
                trainer.Prepare( info );
                const MatrixBase& output = trainer.Compute( input );

                if ( !info.nce ) {
                    loss += output.Xent( target );
                }
                
                exampleCnt += input.Rows();

                if ( input.Rows() == batchSize ) {
                    trainer.BackPropagate( target, lr );
                }

                long current = time( NULL );
                if ( current - lastReport >= 300 ) {
                    cout << CurrentTime() << ")   " 
                         << exampleCnt << " examples passed" << endl;
                    lastReport = current;
                }
            }

            if ( n_noise == 0.0f ) {
                double avgLoss = loss / (double)exampleCnt;
                cout << right
                     << CurrentTime() << ") average cross-entropy loss of " << exampleCnt
                     << " training examples: " << KGRN << avgLoss << KNRM << " or "
                     << KGRN << exp(avgLoss) << KNRM << " in PPL " << endl;
            }

            if ( NULL != test ) {
                float loss = 0.0f;
                int exampleCnt = 0;
                test->Reset();

                while ( test->HasNext() ) {
                    test->PrepareNext();
                    SubMatrix input = test->GetInput();
                    SubMatrix target = test->GetTarget();

                    ExtraInfo info( input.Rows(), 
                                    test->GetSentenceLength(),
                                    false,
                                    test->GetTargetAndNoise(),
                                    unigram,
                                    test->GetPosition() );
                    trainer.Prepare( info );
                    const MatrixBase& output = trainer.Compute( input );
                    loss += output.Xent( target );
                    exampleCnt += input.Rows();
                }

                double avgLoss = loss / (double)exampleCnt;
                cout << right
                     << CurrentTime() << ") average cross-entropy loss of " << exampleCnt
                     << " test examples: " << KGRN << avgLoss << KNRM << " or "
                     << KGRN << exp(avgLoss) << KNRM << " in PPL " << endl;
            }

            if ( NULL != valid ) {
                float loss = 0.0f;
                int exampleCnt = 0;
                valid->Reset();

                while ( valid->HasNext() ) {
                    valid->PrepareNext();
                    SubMatrix input = valid->GetInput();
                    SubMatrix target = valid->GetTarget();
                    
                    ExtraInfo info( input.Rows(), 
                                    valid->GetSentenceLength(),
                                    false,
                                    valid->GetTargetAndNoise(),
                                    unigram,
                                    valid->GetPosition() );
                    trainer.Prepare( info );
                    const MatrixBase& output = trainer.Compute( input );
                    loss += output.Xent( target );
                    exampleCnt += input.Rows();
                }

                double avgLoss = loss / (double)exampleCnt;
                double ppl = exp(avgLoss);
                cout << right
                     << CurrentTime() << ") average cross-entropy loss of " << exampleCnt
                     << " validation examples: " << KGRN << avgLoss << KNRM << " or "
                     << KGRN << ppl << KNRM << " in PPL " << endl;

                if ( !lrDecayStarted ) {
                    if ( ppl + 1.0f > prevPPL ) {
                        lr *= 0.5f;
                        maxIter = epoch + 6;
                        lrDecayStarted = true;
                        trainer.LoadParam( oss.str().c_str() );
                        // trainer.LoadTrainingBuffer( (oss.str() + ".buffer").c_str() );
                        epoch--;
                    }
                    prevPPL = ppl;
                }
                else {
                    lr *= 0.5f;
                }
            }

            cout << right
                 << CurrentTime() << ") epoch" 
                 << setw(3) << epoch << " done" << endl; 
            trainer.Report();
        }

        if ( NULL != train ) {
            delete train;
            train = NULL;
            cout << CurrentTime() << ") training set released" << endl;
        }

        if ( NULL != valid ) {
            delete valid;
            valid = NULL;
            cout << CurrentTime() << ") validation set released" << endl;
        }

        if ( NULL != test ) {
            delete test;
            test = NULL;
            cout << CurrentTime() << ") test set released" << endl;
        }

        return EXIT_SUCCESS;
    }

    else {
        printf( KRED );
        printf( "Usage: %s <config> <basename> <learning-rate> <batch-size> <order> [noise-number] \n", argv[0] );
        printf( "    <config>        : network architecture and configuration \n");
        printf( "    <basename>      : <basename>.{config, vocab, train.numeric, valid.numeric, test.numeric} \n" );
        printf( "    <learning-rate> : initial global learning rate \n" );
        printf( "    <batch-size>    : size of a minibatch \n" );
        printf( "    <order>         : size of context window \n" );
        printf( "    <noise-number>  : number of noise in NCE \n" );
        printf( KNRM );
        exit( EXIT_FAILURE );
    }
}
