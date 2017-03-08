/*
 * simplegraph.cpp
 *
 *  Created on: 26 Feb 2014
 *      Author: time
 */

#include "simplegraph.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
//#include <set>

using namespace std;

// Vector flattener from http://stackoverflow.com/questions/38874605/generic-method-for-flattening-2d-vectors

template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &orig)
{   
    std::vector<T> ret;
    for(const auto &v: orig)
        ret.insert(ret.end(), v.begin(), v.end());                                                                                         
    return ret;
}



// Constructor
simplegraph::simplegraph(){
	}

	/*
	 * Adds a new vertex with no stubs.  
	 * Vertices are given an index starting from 0
	 * Returns the index of new vertex (= number of vertices -1)
	 */
	int
	simplegraph::addVertex(){
	    //v2v.push_back(set<int>());
		//vector<int> blankvector;
		v2v.push_back(vector<int>() );
		return v2v.size()-1;
	}

	

	/**
	 * Adds a new edge from existing vertices s and t.
	 * No checks on vertex indices performed.
	 * No check for multiple edges of self-loops.
	 */
	int
	simplegraph::addEdge(int s, int t){
		if (std::find(v2v[s].begin(), v2v[s].end(),t)!=v2v[s].end()){
			return 0;
		} else{
			v2v[s].push_back(t);
			v2v[t].push_back(s);
		
			stubs.push_back(s);
			stubs.push_back(t);
			
			return 1;
		}
	}
	
	
	/**
	 * Returns the vertex that stub i is connected to.
	 */
	int
	simplegraph::getStub(int i) {
		return stubs.at(i);
	}
	

	/**
	 * Adds a new edge from existing vertices s and t.
	 * Checks on vertex indices and increases size if needed.
	 * Will not add self-loops.
	 * Muliple edges may be added depedning on behaviour of addedge(s,t) method.
	 * Returns negative if self-loop attempted, 0 if OK.
	 */
	int
	simplegraph::addEdgeSlowly(int s, int t){
		if (s==t) return -1;
		int maxSize=max(s,t)+1;
		if (v2v.size()<maxSize){
			v2v.resize(maxSize);
		}
		addEdge(s,t);
		return 0;
	}
	/**
	 * Fetches the n-th neighbour of vertes s
	 * Input
	 * s index of vertex whose neighhbour is required
	 * n require n-th neighbour
	 * Returns
	 * index of neighbour
	 */
	int
	simplegraph::getNeighbour(int s, int n){
		return v2v[s][n]; // is notation correct?
	};

	/**
	 * Returns the total number of stubs.
	 * Stubs are one end of a vertex so this is double the number of edges.
	 * Note that this is currently slow as it sums the degrees.
	 * Would be better to maintain a running total updated when addedge is called.
	 */
	int
	simplegraph::getNumberStubs(){
		return stubs.size();
	};

	/**
	 * Returns the total number of edges.
	 * Note that this is currently slow as it sums the degrees.
	 * Would be better to maintain a running total updated when addedge is called.
	 */
	int
	simplegraph::getNumberEdges(){
		return getNumberStubs()/2;
	};

	/**
	 * Returns number of vertices.
	*/
	int
	simplegraph::getNumberVertices(){
		return v2v.size();
	};

	/**
		 * Returns the degree of vertex s
		 * Input
		 * s index of vertex whose neighhbour is required
		 * Returns
		 * degree of vertex
		 */
	int
	simplegraph::getVertexDegree(int s){
		return v2v[s].size();
	};

	/**
	* Takes the degree distribution as an argument,
	* and adds the distribution of the current graph to it.
	* Also returns the maximum degree k for the current graph.
	* int k_max = getDegreeDistribution(dd)
	* then dd[k] is the number of vertices of degree[k]
	* and (dd.size()-1) is the largest degree in the network
	*/
	int
	simplegraph::getDegreeDistribution(vector<int>  & dd){
		//vector<int> dd;
		int max_k = 0;
		for (int v=0; v<getNumberVertices(); v++){	
			int k= getVertexDegree(v);
			if (dd.size()<=k){
				dd.resize(k+1,0); // vector must have size equal to maximum degree stored +1 as starts from k=0
			}
			if (k>max_k) {
				max_k = k;
			}
			dd[k]++;
		}
		return max_k;
	}

	
	void
	simplegraph::write(char *outFile) {
	  ofstream fout(outFile);
	  if (!fout) {
	      cerr << "Can't open output file " << outFile << endl;
	      exit(1);
	  }
	  write(fout) ;
	  fout.close();
	}

	/**
	 * Output tab separated edge list of graph with no header line.
	 * Useful for some packages.
	 */
	void
	simplegraph::write(ostream & fout) {write(fout, false);}

	/**
	 * Output tab separated edge list of graph with optional header line.
	 * Needs ostream for output (e.g. cout) and true (false) if want header line
	 */
	void
	simplegraph::write(ostream & fout, bool labelOn) {
		if (labelOn) {
			fout << "v1 \t v2" << endl;
		}
		int source,target=-1;
		for (int source=0; source<getNumberVertices(); source++)
			for (int n=0; n<getVertexDegree(source); n++)
			{
				target = getNeighbour(source, n); 
				if (source<target) {
					fout << source << "\t" << target << endl;
				}
			}
	}