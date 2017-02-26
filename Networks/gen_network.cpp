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
#include <string>
#include <cstring>
#include <iomanip>
#include <random> /* Needed to generate large random numbers */
#include <fstream> /* Needed for file output */
//#include <set>

using namespace std;


int main(int argc, char *argv[]) {

	cout << "Arguments are: N m where arguments are separated by spaces and put after command name" << endl; 
	cout << "               N  = number of vertices" << endl; 
	cout << "               m = Number of edges added at each step." << endl; 

// *************************************************************
// User defined variables - default values then command line values are processed

    // Number of vertices
	int N=10;

    // Number of edges added at each step.
	int m=3;

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
		default:
			cout << " Too many arguments" << endl;
		}
	}

	
	// for use of simple built in C random number generator see
    // http://www.cplusplus.com/reference/cstdlib/rand/
	/* initialize random seed: */
		
	// srand(0); // use the fixed seed for debugging
		
	srand (time(NULL)); // Use the time version for real runs

	
   // start by defining an empty graph 
	simplegraph g;
	
	// Initiate complete graph with m vertices.
	for (int v=0; v<m; v++){
		g.addVertex();
		for (int i=0; i<v; i++){
			g.addEdge(v, i);
		}
	}
	
	// Taken from http://stackoverflow.com/questions/28909982/generate-random-number-bigger-than-32767
	// Generates large random numbers (rand() only goes up to 16 bit integers).
	std::random_device rd;
	std::default_random_engine eng {rd()};
	std::uniform_int_distribution<> dist(0, 6*N);
	
	
	for (int v=m; v<N; v++){
		g.addVertex();
		int temp = g.getNumberStubs();
		while (g.getNumberStubs() < temp + 2*m){
			int i = dist(eng) % temp;
			int t = g.getStub(i);
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
	std::string edge_str = "./data/edgelist_";
	std::string m_str = std::to_string(m);
	std::string underscore = "_";
	std::string N_str = std::to_string(N);
	std::string end = ".txt";
	edge_str.append(m_str);
	edge_str.append(underscore);
	edge_str.append(N_str);
	edge_str.append(end);
	
	// Adapted from http://stackoverflow.com/questions/7352099/stdstring-to-char .
	char *edge_char = new char[edge_str.length() + 1];
	strcpy(edge_char, edge_str.c_str());
	
	g.write(edge_char);


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
    //ofstream fout("c:\DATA\CandN\degreediFstribution.dat");
	std::string dist_str = "./data/degreedist_";
	dist_str.append(m_str);
	dist_str.append(underscore);
	dist_str.append(N_str);
	dist_str.append(end);
	
	char *dist_char = new char[dist_str.length() + 1];
	strcpy(dist_char, dist_str.c_str());
	
	ofstream fout(dist_char);
	fout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		fout << k << " \t " << dd[k] << endl; 
	}
	fout.close();

	return 0;
}
