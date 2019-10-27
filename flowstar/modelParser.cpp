/*---
  Flow*: A Verification Tool for Cyber-Physical Systems.
  Authors: Xin Chen, Sriram Sankaranarayanan, and Erika Abraham.
  Email: Xin Chen <chenxin415@gmail.com> if you have questions or comments.
  
  The code is released as is under the GNU General Public License (GPL).
---*/

#include "modelParser.h"
#include "DNN.h"

extern int yyparse();

std::string dnn::DNN_Filename;
extern bool dnn::dnn_initialized;
extern bool dnn::plant_reset;
extern std::vector<std::string> dnn::initialConds;
//extern bool dnn::load_reset;
//extern float dnn::totalNumBranches;
//extern float dnn::dnn_runtime;

int main(int argc, const char *argv[])
{
        if(argc > 1) {
	        dnn::DNN_Filename = argv[1];
	}
	
        dnn::dnn_initialized = false;
	dnn::plant_reset = false;
	
	yyparse();

	printf("total branches: %d\n", dnn::totalNumBranches);
	printf("dnn runtime: %f\n", dnn::dnn_runtime);

	printf("\nInitial conditions:\n");
	for(int i = 0; i < dnn::initialConds.size(); i++){
	        printf("%s\n", dnn::initialConds[i].c_str());
	}

	return 0;
}



