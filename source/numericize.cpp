/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : numericize.cpp
Last Update : Feb 2, 2016
Description : Map the data set into word indecies
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// g++ -o ../numericize numericize.cpp -O3 -rdynamic

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
#include <cctype>
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
        ifstream vocab( argv[1] );    // word list
        ASSERT( vocab.is_open() );

        ifstream input( argv[2] );    // text file
        ASSERT( input.is_open() );

        ofstream output( argv[3], ios::binary );
        ASSERT( output.is_open() );

        map<string,float> word2idx;

        string line;
        while ( getline( vocab, line ) ) {
            line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
            if ( line.size() == 0 ) {
                continue;
            }
            else {
                transform( line.begin(), line.end(), line.begin(), ::tolower );
            }
            istringstream iss( line );
            string word;
            iss >> word;
            
            // Don't write this in one line; it's different
            // word2idx[word] = (float) word2idx.size();
            float idx = (float) word2idx.size();
            word2idx[word] = idx;
            // cout << word << "   " << word2idx[word] << endl;
        }
        vocab.close();


        if ( word2idx.find("<s>") == word2idx.end() ) {
            float idx = (float)word2idx.size();
            word2idx["<s>"] = idx;
        }    // @xmb20160303 the word list may not have these 3 special tokens

        if ( word2idx.find("</s>") == word2idx.end() ) {
            float idx = (float)word2idx.size();
            word2idx["</s>"] = idx;
        }    // @xmb20160303

        if ( word2idx.find("<unk>") == word2idx.end() ) {
            float idx = (float)word2idx.size();
            word2idx["<unk>"] = idx;
        }    // @xmb20160303

        float idx = word2idx["<s>"];
        output.write( (char*)&idx, sizeof(float) );
        cout << "(<s>," << idx << ")  "; 

        idx = word2idx["</s>"];
        output.write( (char*)&idx, sizeof(float) );
        cout << "(</s>," << idx << ")" << endl;
        
        #ifdef RESTRAINT_DEBUG
            int restraint_cnt = 0;
        #endif

        line.clear();
        string token;
        int length;
        while( getline( input, line ) ) {
            line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
            transform( line.begin(), line.end(), line.begin(), ::tolower );
            istringstream iss( line );
            vector<float> index;

            while ( iss >> token ) {
            #ifdef RESTRAINT_DEBUG
                if ( token == "Restraint" || token == "restraint" ) {
                    if ( token == "Restraint" ) {
                        cout << "mingbin is hopeless" << endl;
                    }
                    restraint_cnt++;
                }
            #endif
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

        #ifdef RESTRAINT_DEBUG
            cout << "there are " << restraint_cnt << " 'restraint' in the file" << endl;
        #endif
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
