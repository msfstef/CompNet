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

	cout << "Arguments are: N m (runs) (method) (L)" << endl; 
	cout << "               N  = number of vertices" << endl; 
	cout << "               m = Number of edges added at each step." << endl; 
	cout << "               runs = Number of total runs (default = 1)." << endl; 
	cout << "               method = 0 - Preferential Attachment (default)." << endl; 
	cout << "               		1 - Random Attachment" << endl; 
	cout << "               		2 - Random Walk" << endl; 
	cout << "               L = Length of random walk for method 3 (default = 0)." << endl; 

// *************************************************************
// User defined variables - default values then command line values are processed

    // Number of vertices
	int N=10;

    // Number of edges added at each step.
	int m=3;
	
	// Number of total runs.
	int runs=1;
	
	// Method with which to attach edges.
	int method=0;
	
	// Length of random walk, if random walk attachment is chosen.
	int L=0;

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
		case 4: 
			method= atoi(argv[4]);
			break;
		case 5: 
			L = atoi(argv[5]);
			break;
		default:
			cout << " Too many arguments" << endl;
		}
	}
	
	// Defining string values for file saving.
	string m_str = to_string(m);
	string N_str = to_string(N);
	string method_str = to_string(method);
	string L_str = to_string(L);
	string underscore = "_";
	string end = ".txt";
	
	
	// Defining array for k_max values.
	int k_max [runs];
	
	
	
	// initialize random seed based on system time.
	auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	
	// Generates large random numbers (rand() only goes up to 16 bit integers).
	// Uses mersenne twister algorithm for 32 bit integers.
	mt19937 mt_rand(seed);
	
	// Will create a number of BA models equal to the runs given,
	// to get better statistics for the distribution.
	vector<int> dd;
	for (int run=0; run<runs; run++){
	dd.clear();
	// start by defining an empty graph 
	simplegraph g;
	
	// Initiate complete graph with m vertices.
	for (int v=0; v<m+1; v++){
		g.addVertex();
		for (int i=0; i<v; i++){
			g.addEdge(v, i);
		}
	}
	if (method == 0) {
		// Preferential attachment.
		for (int v=m+1; v<N; v++){
			g.addVertex();
			int stub_no = g.getNumberStubs();
			int connections = 0;
			while (connections < m){
				int i = mt_rand() % stub_no;
				int t = g.getStub(i);
				connections += g.addEdge(v,t);
			}			
		}
	} else if (method == 1) {
		// Pure random attachment.
		for (int v=m+1; v<N; v++){
			g.addVertex();
			int connections = 0;
			while (connections < m){
				int i = mt_rand() % v;
				connections += g.addEdge(v,i);
			}			
		}
	} else if (method == 2) {
		// Random walk attachment.
		for (int v=m+1; v<N; v++){
			g.addVertex();
			int connections = 0;
			while (connections < m){
				int i = mt_rand() % v;
				for (int step; step<L; step++){
					int j = mt_rand() % g.getVertexDegree(i);
					i = g.getNeighbour(i, j);
				}
				connections += g.addEdge(v,i);
			}			
		}
	}
	// Add results to the degree distribution and extract k_max.
	k_max[run] = g.getDegreeDistribution(dd);
	
	
	// Save distribution from run to file.
	string distr_str = "./data/degreedistrun_";
	distr_str.append(m_str);
	distr_str.append(underscore);
	distr_str.append(N_str);
	distr_str.append(underscore);
	distr_str.append(to_string(run));
	distr_str.append(underscore);
	distr_str.append(method_str);
	if (method==3) {
		distr_str.append(underscore);
		distr_str.append(L_str);
	}
	distr_str.append(end);
	
	// Adapted from http://stackoverflow.com/questions/7352099/stdstring-to-char .
	char *distr_char = new char[distr_str.length() + 1];
	strcpy(distr_char, distr_str.c_str());

	ofstream frout(distr_char);
	frout << "k \t n(k)" << endl;
	for (int k=0; k<dd.size(); k++){
		frout << k << " \t " << dd[k] << endl; 
	}
	frout.close();
	
	
	// Will only run for individual run, for debugging purposes.
	if (runs==1){
 	cout << "Network has " << g.getNumberVertices() << " vertices" << endl;
	cout << "Network has " << g.getNumberEdges() <<  " edges" << endl;
	
	// Write out list of edges to file.
	string edge_str = "./data/edgelist_";
	edge_str.append(m_str);
	edge_str.append(underscore);
	edge_str.append(N_str);
	edge_str.append(end);
	
	char *edge_char = new char[edge_str.length() + 1];
	strcpy(edge_char, edge_str.c_str());
	
	g.write(edge_char);
	}
	}
	
	
	/*
	// Write degree distribution to a file.
	// This declares fout to be like cout but everything sent to fout goes to the file
	// named in the declation
    //ofstream fout("c:\DATA\CandN\degreediFstribution.dat");
	string dist_str = "./data/degreedist_";
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
	fout.close(); */
	
	
	// Write k_max values to file.
	string kmax_str = "./data/kmax_";
	kmax_str.append(m_str);
	kmax_str.append(underscore);
	kmax_str.append(N_str);
	kmax_str.append(underscore);
	kmax_str.append(method_str);
	if (method==3) {
		kmax_str.append(underscore);
		kmax_str.append(L_str);
	}
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