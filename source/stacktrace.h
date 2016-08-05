/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Last Update : Feb 2, 2016
Filename    : stacktrace.h
Description : Retrieve the call stack in a human readable way

Copyright (c) belongs to iNCML, https://wiki.eecs.yorku.ca/lab/MLL/
 */


#ifndef STACKTRACE_H_INCLUDED
#define STACKTRACE_H_INCLUDED

#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <cxxabi.h>
using namespace std;


static inline void print_stacktrace( FILE *out = stderr, unsigned int max_frames = 63 ) {
    fprintf(out, "stack trace:\n");

    void* addrlist[max_frames + 1];

    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if ( addrlen == 0 ) {
        fprintf(out, "  <empty, possibly corrupt>\n");
        return;
    }

    char** symbollist = backtrace_symbols(addrlist, addrlen);

    size_t funcnamesize = 256;
    char* funcname = (char*)malloc(funcnamesize);


    for (int i = 1; i < addrlen; i++) {
        char* begin_name = 0;
        char* begin_offset = 0;
        char* end_offset = 0;

        for (char *p = symbollist[i]; *p; ++p) {
            if ( *p == '(' ) {
                begin_name = p;
            }
            else if ( *p == '+' ) {
                begin_offset = p;
            }
            else if ( *p == ')' && begin_offset ) {
                end_offset = p;
                break;
            }
        }

        if ( begin_name && begin_offset && end_offset
             && begin_name < begin_offset ) {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            int status;
            char* ret = abi::__cxa_demangle(begin_name,
                            funcname, &funcnamesize, &status);
            if (status == 0) {
                funcname = ret;
                fprintf(out, "  %s : %s + %s\n", symbollist[i], funcname, begin_offset);
            }
            else {
                fprintf(out, "  %s : %s() + %s\n", symbollist[i], begin_name, begin_offset);
            }
        }
        else {
            fprintf(out, "  %s\n", symbollist[i]);
        }
    }

    free( funcname );
    free( symbollist );
}

#endif
