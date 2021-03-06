//
//  main.cpp
//  NeuralNet
//
//  Created by Erik Arakelyan on 5/10/16.
//  Copyright © 2016 Erik Arakelyan. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <math.h>
#include <fstream>
#include <sstream>

#include "NeuralNet.hpp"
#include "TrainingData.hpp"

using namespace std;







//__________________________ Interface For testing ___________________________________


void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    
    cout << endl;
}



int main() {
    
    /*  simple initial tester
     
     
    vector <unsigned> Constructoptions; //specify the net structure EX:3-2-1
    Constructoptions.push_back(3);
    Constructoptions.push_back(2);
    Constructoptions.push_back(1);
    
    
    NeuralNet Net(Constructoptions);//number of layers, and neurons in each layer
    
    vector <double> inputValues;
    
    Net.fforward(inputValues);
    
    vector <double> targetedValues;
    
    Net.backprop(targetedValues);
    
    vector <double> retriveValues;
    Net.retrieveResults(retriveValues);
    
    */
    
    
    
     /* XOR training for neural network
    
    TrainingData trainData("/Users/Erik/Desktop/NeuralNet/NeuralNet/train.txt");
    
    // e.g., { 3, 2, 1 }
    vector<unsigned> Constructoptions;
    trainData.getTopology(Constructoptions);
    
    NeuralNet myNet(Constructoptions);
    
    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    
    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;
        
        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != Constructoptions[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.fforward(inputVals);
        
        // Collect the net's actual output results:
        myNet.retrieveResults(resultVals);
        showVectorVals("Outputs:", resultVals);
        
        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == Constructoptions.back());
        
        myNet.backprop(targetVals);
        
        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
        << myNet.getRecentAverageError() << endl;
    }
    
    cout << endl << "Done" << endl;
    */
    
    
    
  

    
    vector<vector<double> > values;
    vector<double> valueline;
    ifstream fin("/Users/Erik/Desktop/Simple NN/Simple-NN/NeuralNet/SACdata.csv");
    string item;
    for (string line; getline(fin, line); )
    {
        istringstream in(line);
        
        while (getline(in, item, ','))
        {
            valueline.push_back(atof(item.c_str()));
        }
        
        values.push_back(valueline);
        valueline.clear();
    }
    
    
    vector<unsigned> Constructoptions;
    Constructoptions.push_back(values[1].size() - 3);
//    Constructoptions.push_back(3);
//    Constructoptions.push_back(9);
//    Constructoptions.push_back(27);
    Constructoptions.push_back(1);
    Constructoptions.push_back(2);
    Constructoptions.push_back(4);
    Constructoptions.push_back(4);
    Constructoptions.push_back(4);



   // Constructoptions.push_back(rand()%50);
   // Constructoptions.push_back(rand()%50);

//    Constructoptions.push_back(27);
//    Constructoptions.push_back(9);
//    Constructoptions.push_back(3);


//    Constructoptions.push_back(values[1].size() - 3);
//    Constructoptions.push_back(values[1].size() - 3);
//    Constructoptions.push_back(values[1].size() - 3);
    
    Constructoptions.push_back(1);

    
    NeuralNet SACNet(Constructoptions);

    // 50 - 5 - 1 Neuralnet


    
    cout<<values.size()<<endl;
    cout<<values[1].size()<<endl;
    
    for (int epoch = 1 ; epoch <= values.size() - 2 ; epoch++)
    {
        vector<double> inputVals, targetVals, resultVals;

        //giving the input values to the net
        for (unsigned long input = 0; input<values[epoch].size() - 3; input++)
        {
            inputVals.push_back(values[epoch][input]);
        }
        
        //giving the targeted values to the net
        for (unsigned long target = values[epoch].size() - 1; target<values[epoch].size(); target++)
        {
            if(values[epoch][target] >= 14 )
            {
                targetVals.push_back(1);
            }
            else
            {
                targetVals.push_back(0);
            }
        }
        
        
        SACNet.fforward(inputVals);
        
        // Collect the net's actual output results:
        SACNet.retrieveResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        
        // Train the net what the outputs should have been:
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == Constructoptions.back());
        
        SACNet.backprop(targetVals);
        
        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
        << SACNet.getRecentAverageError() << endl;
        
        
        cout<<"epoch "<<epoch<<" ended"<<endl;
        cout<<endl;

    }
    
    
    // Now lets test the net on ME :)
    // I already added myself scaled to the last row of the dataset
    
    /*
    vector<double> inputVals,resultVals;
    //giving the input values to the net
    for (unsigned long input = 0; input<values[values.size() - 1].size() - 3; input++)
    {
        inputVals.push_back(values[values.size() - 1][input]);
    }
    
    SACNet.fforward(inputVals);
    // Collect the net's actual output results:
    SACNet.retrieveResults(resultVals);
    showVectorVals("Outputs:", resultVals);
    
    
    // i passed the course really :) with A
    cout<< "Target is 0"<<endl;
    
    cout << "Net recent average error: "
    << SACNet.getRecentAverageError() << endl;
     */
    
    
    return 0;
}