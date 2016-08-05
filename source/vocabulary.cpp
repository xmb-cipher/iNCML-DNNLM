/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : vocabulary.cpp
Last Update : Feb 2, 2016
Description : Retrieve a word list of a given training sets
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// g++ -o ../vocabulary vocabulary.cpp -O3 -rdynamic

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
        int minCount = atoi( argv[2] );
        ASSERT( minCount > 0 );

        int maxRemained = atoi( argv[3] );
        ASSERT( maxRemained > 3 );

        ifstream file;
        file.open( argv[1] );
        ASSERT( file.is_open() );

        map<string,int> word2num;
        word2num[string("</s>")] = 0;
        string line;
        string token;

        while ( getline( file, line ) ) {
            line.erase( line.find_last_not_of( " \n\r\t" ) + 1 );
            transform( line.begin(), line.end(), line.begin(), ::tolower );
            istringstream iss( line );
            while ( iss >> token ) {
                if ( word2num.find(token) != word2num.end() ) {
                    word2num[token]++;
                }
                else {
                    word2num[token] = 1;
                }
            }
            word2num["</s>"]++;
        }
        file.close();
        
        int n_eos = word2num["</s>"];
        int n_unk = word2num.find("<unk>") == word2num.end() ? 0 : word2num["<unk>"];
        word2num.erase( "<unk>" );
        word2num.erase( "</s>" );

        map<int,vector<string> > num2word;
        for ( map<string,int>::iterator itr = word2num.begin(); itr != word2num.end(); itr++ ) {
            if ( itr->second >= minCount ) {
                if ( num2word.find(itr->second) == num2word.end() ) {
                    num2word[itr->second] = vector<string>(1, itr->first);
                }
                else {
                    num2word[itr->second].push_back( itr->first );
                }
            }
            else {
                n_unk += itr->second;
            }
        }

        int cnt = 0;
        int acc = 0;
        for ( map<int,vector<string> >::reverse_iterator itr = num2word.rbegin();
              itr != num2word.rend(); itr++ ) {
            for ( int i = 0; i < itr->second.size(); i++ ) {
                acc += itr->first;
                if ( cnt < maxRemained - 3 ) {
                    cout << setw(16) << itr->second[i] << "  "
                         << setw(10) << cnt << "  "
                         << setw(10) << acc << endl;
                    cnt++;
                }
                else {
                    n_unk += itr->first;
                }
            }
        }

        acc++;
        cout << setw(16) << "<s>" << "  " 
             << setw(10) << cnt << "  "
             << setw(10) << acc << endl;
        cnt++;

        acc += n_eos;
        cout << setw(16) << "</s>" << "  " 
             << setw(10) << cnt << "  "
             << setw(10) << acc << endl;
        cnt++;

        acc += n_unk;
        cout << setw(16) << "<unk>" << "  " 
             << setw(10) << cnt << "  "
             << setw(10) << acc << endl;

        return EXIT_SUCCESS;
    }


    else {
        printf ( KRED );
        printf( "Usage: %s <training-set> <min-count> <size-limit>\n", argv[0] );
        printf( "    <training-set> : text file, one sentence per line\n" );
        printf( "    <min-count>    : words whose occurrence is less than <min-count> is mapped to <unk>\n" );
        printf( "    <size-limit>   : only the first <size-limit> words are kept\n" );
        printf( KNRM );
        return EXIT_FAILURE;
    }
}
