//
//  NeuralNet.cpp
//  NeuralNet
//
//  Created by Erik Arakelyan on 5/11/16.
//  Copyright © 2016 Erik Arakelyan. All rights reserved.
//

#include "NeuralNet.hpp"

void NeuralNet:: retrieveResults(vector<double> &retriveValues) const
{
    retriveValues.clear();
    
    LayerOfNet lastLayer = layers.back();
    
    for(unsigned n = 0; n<lastLayer.size() - 1; n++)
    {
        retriveValues.push_back(lastLayer[n].getOutputValue());
    }
}

void NeuralNet::fforward(const vector<double> &inputValues)
{
    assert(inputValues.size() == layers[0].size() - 1); //check
    
    
    //initializ
    for(int i = 0; i < inputValues.size(); i++ )
    {
        layers[0][i].setOutputValue(inputValues[i]); // give the initial-neurons layer the input
        
    }
    
    
    //Forward propogation
    
    for(unsigned layerNum = 1; layerNum <layers.size();layerNum++)
    {
        LayerOfNet &previousLayer = layers[layerNum-1];
        for(unsigned neuronNum = 0; neuronNum <layers[layerNum].size() - 1;neuronNum++)
        {
            layers[layerNum][neuronNum].feedForward(previousLayer); //preactivating using activation of previous neurons + activating
        }
        
    }
    
}


void NeuralNet::backprop(const vector<double> &targetedValues)
{
    // 1) calculate log-loss or RMS
    // 2) calculate output layer gradients
    // 3) calculate hidden layer gradients
    // 4) update all the weights and biases of all layers
    
    
    
    // 1. calculate log-loss or RMS
    
    LayerOfNet &outputLayer = layers.back();
    netError = 0.0;
    
    for(unsigned n = 0; n < outputLayer.size() - 1 ;n++)
    {
        double delta = targetedValues[n] - outputLayer[n].getOutputValue();
        netError += delta * delta;
    }
    
    netError /= outputLayer.size() - 1;
    netError =  sqrt(netError);
    
    // Recent average Calculation
    
    recentAverageNetError =
    (recentAverageNetError*recentAverageSmoothingFactor+netError)/(recentAverageSmoothingFactor+1.0);
    
    
    // 2. calculate output layer gradients
    
    for(unsigned n = 0; n < outputLayer.size() - 1 ; n++)
    {
        outputLayer[n].calaculateOutputGradient(targetedValues[n]);
    }
    
    
    // 3. calculate hidden layer gradients
    
    for(unsigned layerNum = layers.size() - 2 ; layerNum > 0 ; layerNum--)
    {
        LayerOfNet &currentHiddenLayer = layers[layerNum];
        LayerOfNet &nextLayer = layers[layerNum + 1];
        
        for(unsigned n = 0; n < currentHiddenLayer.size() - 1 ; n++)
        {
            currentHiddenLayer[n].calculateHiddenGradients(nextLayer); //we need values of preActivation in k+1 layer for derivative in k-th layer
            
        }
    }
    
    // 4. update all the weights and biases of all layers
    
    for(unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--)
    {
        LayerOfNet &currentLayer = layers[layerNum];
        LayerOfNet &previousLayer = layers[layerNum - 1];
        for(unsigned n = 0; n < currentLayer.size() - 1 ; n++)
        {
            currentLayer[n].updateInputWeights(previousLayer);
        }
        
    }
    
}

double NeuralNet::softmax(vector<double> &outputValues) {
    double maximum = outputValues[0], denominator = 0, current = outputValues[0];
    for(vector<double>::iterator it = outputValues.begin(); it != outputValues.end(); ++it) {
        current = *it;
        if(maximum < current)
            maximum = current;
        denominator += exp(current);
    }
    return exp(maximum) / denominator;
}

NeuralNet::NeuralNet(vector <unsigned> &Constructoptions)
{
    unsigned numLayers = Constructoptions.size();
    for(unsigned layerCount=0;layerCount<numLayers;layerCount++)
    {
        layers.push_back(LayerOfNet()); //making of a new layer
        
        unsigned numberOfOutputs = layerCount ==Constructoptions.size()-1  ?  0: Constructoptions[layerCount+1]; // checking if were on output layer or not
        
        //After adding the layer lets fill it with Neurons
        for(unsigned neuronNum = 0; neuronNum <= Constructoptions[layerCount];neuronNum++) // <= because we have a bias
        {
            layers.back().push_back(Neuron(numberOfOutputs,neuronNum)); //taking the last added layer with .back
            cout<<"Made a Neuron"<<endl;
        }
        
        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        layers.back().back().setOutputValue(1.0);
        
    }
    
    
}
