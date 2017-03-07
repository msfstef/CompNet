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
#include <chrono> /* Needed for better seeding of special generator */
#include <random> /* Needed to generate large random numbers */
#include <fstream> /* Needed for file output */
//#include <set>

using namespace std;


int main(int argc, char *argv[]) {

	cout << "Arguments are: N m (runs)" << endl; 
	cout << "               N  = number of vertices" << endl; 
	cout << "               m = Number of edges added at each step." << endl; 
	cout << "               runs = Number of total runs (default = 1)." << endl; 

// *************************************************************
// User defined variables - default values then command line values are processed

    // Number of vertices
	int N=10;

    // Number of edges added at each step.
	int m=3;
	
	// Number of total runs.
	int runs=1;

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
			runs= atoi(argv[3]);
			break;
		default:
			cout << " Too many arguments" << endl;
		}
	}
	
	// Defining string values for file saving.
	std::string m_str = std::to_string(m);
	std::string underscore = "_";
	std::string N_str = std::to_string(N);
	std::string end = ".txt";
	
	// Defining array for k_max values.
	int k_max [runs];
	
	
	
	// initialize random seed based on system time.
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	
	// Taken from http://stackoverflow.com/questions/28909982/generate-random-number-bigger-than-32767
	// Generates large random numbers (rand() only goes up to 16 bit integers).
	std::default_random_engine eng {seed};
	std::uniform_int_distribution<> dist(0, 2*m*N);
	
	// Will create a number of BA models equal to the runs given,
	// to get better statistics for the distribution.
	vector<int> dd;
	for (int run=0; run<runs; run++){
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
		int temp = g.getNumberStubs();
		while (g.getNumberStubs() < temp + 2*m){
			int i = dist(eng) % temp;
			int t = g.getStub(i);
			g.addEdge(v,t);
		}			
	}
	// Add results to the degree distribution.
	k_max[run] = g.getDegreeDistribution(dd);

	
	
	if (runs==1){
 	cout << "Network has " << g.getNumberVertices() << " vertices" << endl;
	cout << "Network has " << g.getNumberEdges() <<  " edges" << endl;
	
	// Write out list of edges to file.
	std::string edge_str = "./data/edgelist_";
	edge_str.append(m_str);
	edge_str.append(underscore);
	edge_str.append(N_str);
	edge_str.append(end);
	
	// Adapted from http://stackoverflow.com/questions/7352099/stdstring-to-char .
	char *edge_char = new char[edge_str.length() + 1];
	strcpy(edge_char, edge_str.c_str());
	
	g.write(edge_char);
	}
	}
	
    /* // output on screen
    cout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		cout << k << " \t " << dd[k] << endl;  
	}*/
	
	
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
	
	
	// Write k_max values to file.
	std::string kmax_str = "./data/kmax_";
	kmax_str.append(m_str);
	kmax_str.append(underscore);
	kmax_str.append(N_str);
	kmax_str.append(end);
	
	char *kmax_char = new char[kmax_str.length() + 1];
	strcpy(kmax_char, kmax_str.c_str());
	
	ofstream fout2(kmax_char);
	fout2 << "k_max values" << endl;
	for (int i = 0; i<runs; i++){
		fout2 << k_max[i] << endl; 
	}
	fout2.close();
	
	cout << "Complete." << endl;
	return 0;
}