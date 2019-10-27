#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "Hybrid.h"
#include "Continuous.h"
#include "Polynomial.h"
#include <map>

using namespace flowstar;

namespace dnn {

enum activation {LINEAR = 0, SIGMOID = 1, SWISH = 2, RELU = 3, TANH = 4};
extern std::string DNN_Filename;
extern Continuous_Reachability_Setting dnn_crs;
extern std::vector<ResetMap> dnn_resets;
extern std::vector<activation> dnn_activations;
extern bool dnn_initialized;
extern ResetMap activation_reset;
extern std::map<int, int> branch_origin;
extern std::map<int, Flowpipe> saved_plant_states;
extern bool plant_reset;
extern int totalNumBranches;
extern int curBranchId;
extern float dnn_runtime;
extern bool storedInitialConds;
extern std::vector<std::string> initialConds;

void sig_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars);
  
void tanh_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars);

void relu_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars);

void swish_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars);

void swish10_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars);

void load_dnn(std::vector<ResetMap> &resets, std::vector<activation> &activations, const Variables &vars, const std::string filename = DNN_Filename);

}

#endif /* NEURAL_NETWORK_H_ */
