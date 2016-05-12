//
//  NeuralNet.hpp
//  NeuralNet
//
//  Created by Erik Arakelyan on 5/11/16.
//  Copyright © 2016 Erik Arakelyan. All rights reserved.
//

#ifndef NeuralNet_hpp
#define NeuralNet_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <math.h>
#include <fstream>
#include <sstream>
#include "Neuron.hpp"
using namespace std;
#endif /* NeuralNet_hpp */

// _________________ Class Neural Net_____________________________



class NeuralNet
{
private:
    
    vector <LayerOfNet> layers; //vector of all the layers
    double netError; //loss Function result
    
    double recentAverageNetError; // mijankyal arjeq tvyal pahin
    double recentAverageSmoothingFactor;
    
    
public:
    NeuralNet(vector <unsigned> &Constructoptions);
    
    void fforward(const vector<double> &inputValues);
    void backprop(const vector<double> &targetedValues);
    double softmax(vector<double> &outputValues);
    void retrieveResults(vector<double> &retriveValues) const;
    double getRecentAverageError(void) const { return recentAverageNetError; }
};



