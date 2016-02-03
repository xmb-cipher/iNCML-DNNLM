/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : numericize.cpp
Last Update : Feb 2, 2016
Description : Map the data set into word indecies
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// g++ -o numericize numericize.cpp -O3 -rdynamic

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <vector>
#include <sstream>
#include "stacktrace.h"
using namespace std;


#define ASSERT( status ) __assert( status, __FILE__, __LINE__ );


inline void __assert( bool status, const char* file, int line ) {                               
    if ( !status ) {                
        fprintf( stderr, "%sFAIL at Line %4d in %10s\n%s", KRED, line, file, KNRM );                     
        print_stacktrace();   
        exit( EXIT_FAILURE );                       
    }                                               
}  



int main( int argc, char** argv ) {
	if ( argc == 4 ) {
		ifstream vocab( argv[1] );
		ASSERT( vocab.is_open() );

		ifstream input( argv[2] );
		ASSERT( input.is_open() );

		ofstream output( argv[3], ios::binary );
		ASSERT( output.is_open() );

		map<string,float> word2idx;

		string line;
		while ( getline( vocab, line ) ) {
			line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
			istringstream iss( line );

			string word;
			float idx;
			iss >> word;
			iss >> idx;

			word2idx[word] = idx;
		}
		vocab.close();

		float idx = word2idx["<s>"];
		output.write( (char*)&idx, sizeof(float) );

		idx = word2idx["</s>"];
		output.write( (char*)&idx, sizeof(float) );

		line.clear();
		string token;
		int length;
		while( getline( input, line ) ) {
			line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
			istringstream iss( line );
			vector<float> index;

			while ( !iss.eof() ) {
				iss >> token;
				if ( word2idx.find( token ) == word2idx.end() ) {
					index.push_back( word2idx["<unk>"] );
				}
				else {
					index.push_back( word2idx[token] );
				}
			}

			length = index.size();
			output.write( (char*)&length, sizeof(int) );
			output.write( (char*)&index[0], sizeof(float) * length );
		}
		input.close();
		output.close();
	}


	else {
		printf ( KRED );
		printf( "Usage: %s <vocabulary> <text-file> <numeric-file>\n", argv[0] );
		printf( "    <vocabulary>   : word list with index and accumulatie occurrence\n" );
		printf( "    <text-file>    : text file to be numericized\n" );
		printf( "    <numeric-file> : numericized output in binary\n" );
		printf( KNRM );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}