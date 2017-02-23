/*
 * test.cpp
 *
 * Tests simplegraph and illustrates how to do basic network operations.
 * Generates a basic ER G(N,p) type of random graph to show how to add vertices and edges.
 * Uses basic (rubiish) random number generator built into C++ but that 
 * might be good enough.
 *
 *  Created on: 28 Feb 2014
 *      Author: time
 */
#include "simplegraph.h"
#include <vector>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <iomanip>
#include <fstream> /* Needed for file output */
//#include <set>

using namespace std;


int main(int argc, char *argv[]) {

	cout << "Arguments are: N m (d) where arguments are separated by spaces and put after command name" << endl; 
	cout << "               N  = number of vertices" << endl; 
	cout << "               m = Number of edges added at each step." << endl; 
	cout << "               d  = if letter d is third argument then debugging options will be turned on" << endl; 

// *************************************************************
// User defined variables - default values then command line values are processed

    // Number of vertices
	int N=10;

    // Number of edges added at each step.
	int m=3;

    // if true, then some extra debugging stuff is swicthed on.
    // First random numbers will always be the same every time you run it (fixed seed).
    // Also produces some output on screen, use only for small networks.
    bool debugon = true;

// End of user defined section
// *************************************************************

// First process command line

	for (int a=1; a<argc; a++){
		switch (a) {
		case 1: 
            N=atoi(argv[1]);
			break;
		case 2:
			 m=atoi(argv[2]);
			break;
		case 3: 
			debugon= (*argv[3] == 'd');
			break;
		default:
			cout << " Too many arguments" << endl;
		}
	}

	
	// for use of simple built in C random number generator see
    // http://www.cplusplus.com/reference/cstdlib/rand/
	/* initialize random seed: */
	if (debugon){
		srand(0); // use the fixed seed for debugging
	}
	else{
		srand (time(NULL)); // Use the time version for real runs
	}
	
   // start by defining an empty graph 
	simplegraph g;
	
	// Initiate complete graph with m vertices.
	for (int v=0; v<m; v++){
		g.addVertex();
		for (int i=0; i<v; i++){
			g.addEdge(v, i);
		}
	}
	
	for (int v=m; v<N; v++){
		g.addVertex();
		for (int e=0; e<m; e++){
			int i = rand() % (g.stubs.size()- 2*e);
			int t = g.stubs[i];
			g.addEdge(v,t);
		}			
	}


 	cout << "Network has " << g.getNumberVertices() << " vertices" << endl;
	cout << "Network has " << g.getNumberEdges() <<  " edges" << endl;

	// Write out list of edges to file if you want
	// WARNING with file names and all strings in C++ \ has a special meaning.
	// so for directories on Windows use either \\ for a single backslash or forwards slash / may work
	//g.write("c:/DATA/CandN/edgelist.txt");
	//g.write("c:\\DATA\\CandN\\edgelist.txt");
    // Thie output files will appear in same directory as source code (this file) if you give no directories in filename
    g.write("edgelist.txt");


	// Studying the degree distribution
	vector<int> dd;
	g.getDegreeDistribution(dd);
	
    // output on screen
    cout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		cout << k << " \t " << dd[k] << endl; 
	}

	// Write degree distribution to a file.
	// This declares fout to be like cout but everything sent to fout goes to the file
	// named in the declation
    //ofstream fout("c:\DATA\CandN\degreedistribution.dat");
	ofstream fout("degreedistribution.dat");
	fout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		fout << k << " \t " << dd[k] << endl; 
	}
	fout.close(); 

	return 0;
}
