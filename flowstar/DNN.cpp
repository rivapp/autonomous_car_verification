#include "DNN.h"
#include <sstream>
#include <map>
#include <yaml-cpp/yaml.h>

using namespace YAML;
using namespace flowstar;
using namespace std;

Real sigmoid(Real input){
        Real sig = Real(-1) * Real(input);
	sig.exp_assign();
	
        return Real(1) / (Real(1) + sig);
}

Real tanh(Real input){
        Real posExp = Real(input);
	posExp.exp_assign();

	Real negExp = Real(Real(-1) * input);
	negExp.exp_assign();
	
        return (posExp - negExp)/(posExp + negExp);
}

Real swish(Real input){
        return input * sigmoid(input);
}

Real swishTen(Real input){
        return input * sigmoid(Real(10) * input);
}

Real swishHundred(Real input){
        return input * sigmoid(Real(100) * input);
}

Real lambda(int lorder, int rorder, int order, Real input){
        if (lorder + rorder == order + 1){

	        Real sig = sigmoid(input);
		sig.pow_assign(lorder);

		Real oneMsig = Real(1) - sigmoid(input);
		oneMsig.pow_assign(rorder);

		return sig * oneMsig;
	    
	  
	        //return pow(sigmoid(input), lorder) * pow(1 - sigmoid(input), rorder);
	}

	return Real(lorder) * lambda(lorder, rorder + 1, order, input) -
	  Real(rorder) * lambda(lorder + 1, rorder, order, input);
}

Real tanhlambda(int lorder, int rorder, int rec, int order, Real input){

        Real output;

        if (rec == order - 1){

	  output = Real(1) - tanh(input) * tanh(input);
		output.pow_assign(rorder);
	  
	        //output = pow(1 - tanh(input) * tanh(input), rorder);
        
		if (lorder > 0){
		        Real recReal = tanh(input);
			recReal.pow_assign(lorder);

			output = output * recReal;
			
		        //output = output * pow(tanh(input), lorder);
		}

	}
        
	else{                
	        output =  Real(-1) * Real(2) * Real(rorder) * tanhlambda(lorder + 1, rorder, rec + 1, order, input);
        
		if (lorder > 0)
		        output = output + Real(lorder) * tanhlambda(lorder - 1, rorder + 1, rec + 1, order, input);            
	}

	return output;

}

Real tanhDer(int order, Real input){
        return tanhlambda(0, 1, 0, order, input);
}

Real sigDer(int order, Real input){
        return lambda(1, 1, order, input);
}

Real swish1stDer(Real input){
        int derOrder = 1;
	return sigmoid(input) + input * lambda(1, 1, derOrder, input);
}

Real swish2ndDer(Real input){
        int derOrder = 1;
	return Real(2) * lambda(1, 1, derOrder, input) + input * lambda(1, 1, derOrder + 1, input);
}

Real swish3rdDer(Real input){
        int derOrder = 2;
	return Real(3) * lambda(1, 1, derOrder, input) + input * lambda(1, 1, derOrder + 1, input);
}

Real swishTen1stDer(Real input){
        int derOrder = 1;
	return sigmoid(Real(10) * input) + Real(10) * input * lambda(1, 1, derOrder, Real(10) * input);
}

Real swishTen2ndDer(Real input){
        int derOrder = 1;
	return Real(20) * lambda(1, 1, derOrder, Real(10) * input) + Real(100) * input * lambda(1, 1, derOrder + 1, Real(10) * input);
}

Real swishTen3rdDer(Real input){
        int derOrder = 2;
	return Real(300) * lambda(1, 1, derOrder, Real(10) * input) + Real(1000) * input * lambda(1, 1, derOrder + 1, Real(10) * input);
}

Real swishHundred1stDer(Real input){
        int derOrder = 1;
	return sigmoid(Real(100) * input) + Real(100) * input * lambda(1, 1, derOrder, Real(100) * input);
}

Real swishHundred2ndDer(Real input){
        int derOrder = 1;
	return Real(200) * lambda(1, 1, derOrder, Real(100) * input) + Real(1000) * input * lambda(1, 1, derOrder + 1, Real(100) * input);
}

Real swishHundred3rdDer(Real input){
        int derOrder = 2;
	return Real(30000) * lambda(1, 1, derOrder, Real(100) * input) + Real(1000000) * input * lambda(1, 1, derOrder + 1, Real(100) * input);
}

//The hardcoded bounds used in this function are conservative numerical bounds
Real getSig4thDerBound(Interval bounds){

        //Region 5
        if(bounds.inf() >= 3.15){
	        return sigDer(4, Real(bounds.inf())).abs();
	}

	//Region 4.5
	else if(bounds.inf() >= 3.13 && bounds.inf() <= 3.15){
	        return Real(0.01908);
	}

	//Region 4
        else if(bounds.inf() >= 0.85 && bounds.inf() <= 3.13){

	        Real bound = sigDer(4, Real(bounds.inf())).abs();

		//sup is in Region 4
	        if(bounds.sup() <= 3.13){
		  
		        Real rbound = sigDer(4, Real(bounds.sup())).abs();
			
			if (rbound > bound) bound = rbound;
		}		
		//sup is in Region 5
		else{
		        if (Real(0.01908) > bound) bound = Real(0.01908);
		}
		
	        return bound;
	}

	//Region 3.5
	else if(bounds.inf() >= 0.83 && bounds.inf() <= 0.85){
	        return Real(0.1277);
	}	

	//Region 3
        else if(bounds.inf() >= -0.83 && bounds.inf() <= 0.83){
	  
	        Real bound = sigDer(4, Real(bounds.inf())).abs();

		//sup is in Region 3
	        if(bounds.sup() <= 0.83){
		        Real rbound = sigDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}
		//sup is beyond the global max
		else{
		        bound = Real(0.1277);
		}
		
	        return bound;
	}

	//Region 2.5
	else if(bounds.inf() >= -0.85 && bounds.inf() <= -0.83){
	        return Real(0.1277);
	}

	//Region 2
	else if(bounds.inf() >= -3.13 && bounds.inf() <= -0.85){

	        Real bound = sigDer(4, Real(bounds.inf())).abs();

		//sup is in Region 2
	        if(bounds.sup() <= -0.85){
		        Real rbound = sigDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is beyond global max
		else if(bounds.sup() >= -0.85){
		        bound = Real(0.1277);
		}
		  
	        return bound;
	}

	//Region 1.5
	else if(bounds.inf() >= -3.15 && bounds.inf() <= -3.13){

	        Real bound = Real(0.01908);

		//sup is in Region 2
	        if(bounds.sup() <= -0.85){
		        Real rbound = sigDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is beyond global max
		else if(bounds.sup() >= -0.85){
		        bound = Real(0.1277);
		}
		  
	        return bound;
	}

	//Region 1
	else if(bounds.inf() <= -3.15){

	        Real bound = sigDer(4, Real(bounds.inf())).abs();

		//sup is in Region 1
	        if(bounds.sup() <= -3.15){
		        Real rbound = sigDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is in Region 2
	        if(bounds.sup() <= -0.85){

		        bound = Real(0.01908);
			  
		        Real rbound = sigDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}		

		//sup is beyond global max
		else if(bounds.sup() >= -0.85){
		        bound = Real(0.1277);
		}
		  
	        return bound;
	}	
  
        return Real(0.1277);
}

//The hardcoded bounds used in this function are conservative numerical bounds
Real getSig3rdDerBound(Interval bounds){

        //Region 4
        if(bounds.inf() >= 2.3){
	        return sigDer(3, Real(bounds.inf()));
	}

	//Region 3.5
	else if(bounds.inf() >= 2.28 && bounds.inf() <= 2.3){
	        return Real(0.0417);
	}

	//Region 3
        else if(bounds.inf() >= 0 && bounds.inf() <= 2.28){

	        Real bound = sigDer(3, Real(bounds.inf())).abs();

		//sup is in Region 3
	        if(bounds.sup() <= 2.28){
		  
		        Real rbound = sigDer(3, Real(bounds.sup())).abs();
			
			if (rbound > bound) bound = rbound;
		}		
		//sup is in Region 4
		else{
		        if (Real(0.0417) > bound) bound = Real(0.0417);
		}
		
	        return bound;
	}

	//Region 2
        else if(bounds.inf() >= -2.28 && bounds.inf() <= 0){
	  
	        Real bound = sigDer(3, Real(bounds.inf())).abs();

		//sup is in Region 2
	        if(bounds.sup() <= 0){
		        Real rbound = sigDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		else{
		        bound = Real(0.126);
		}
		
	        return bound;
	}

	//Region 1.5
        else if(bounds.inf() >= -2.3 && bounds.inf() <= -2.28){
	  
	        Real bound = Real(0.0417);

		//sup is in Region 2
	        if(bounds.sup() <= 0){
		        Real rbound = sigDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		else{
		        bound = Real(0.126);
		}
		
	        return bound;
	}				

	//Region 1
	else if(bounds.inf() <= -2.3){

	        Real bound = sigDer(3, Real(bounds.inf())).abs();

		//sup is in Region 1
	        if(bounds.sup() <= -2.3){
		        Real rbound = sigDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is in Region 2
		else if(bounds.sup() >= -2.3 && bounds.sup() <= 0){
		        bound = Real(0.0417);

			Real rbound = sigDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is beyond global max
		else if(bounds.sup() >= 0){
		        bound = Real(0.126);
		}
		  
	        return bound;
	}		
  
        return Real(0.126);
}

//The hardcoded bounds used in this function are conservative numerical bounds
Real getTanh4thDerBound(Interval bounds){

        //Region 5
        if(bounds.inf() >= 1.573){
	        return tanhDer(4, Real(bounds.inf())).abs();
	}

	//Region 4.5
	else if(bounds.inf() >= 1.571 && bounds.inf() <= 1.573){
	        return Real(0.61009);
	}

	//Region 4
        else if(bounds.inf() >= 0.422 && bounds.inf() <= 1.571){

	        Real bound = tanhDer(4, Real(bounds.inf())).abs();

		//sup is in Region 4
	        if(bounds.sup() <= 1.571){
		  
		        Real rbound = tanhDer(4, Real(bounds.sup())).abs();
			
			if (rbound > bound) bound = rbound;
		}		
		//sup is in Region 5
		else{
		        if (Real(0.61009) > bound) bound = Real(0.61009);
		}
		
	        return bound;
	}

	//Region 3.5
	else if(bounds.inf() >= 0.42 && bounds.inf() <= 0.422){
	        return Real(4.0859);
	}	

	//Region 3
        else if(bounds.inf() >= -0.42 && bounds.inf() <= 0.42){
	  
	        Real bound = tanhDer(4, Real(bounds.inf())).abs();

		//sup is in Region 3
	        if(bounds.sup() <= 0.42){
		        Real rbound = tanhDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}
		//sup is beyond the global max
		else{
		        bound = Real(4.0859);
		}
		
	        return bound;
	}

	//Region 2.5
	else if(bounds.inf() >= -0.422 && bounds.inf() <= -0.42){
	        return Real(4.0859);
	}

	//Region 2
	else if(bounds.inf() >= -1.571 && bounds.inf() <= -0.422){

	        Real bound = tanhDer(4, Real(bounds.inf())).abs();

		//sup is in Region 2
	        if(bounds.sup() <= -0.422){
		        Real rbound = tanhDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is beyond global max
		else if(bounds.sup() >= -0.422){
		        bound = Real(4.0859);
		}
		  
	        return bound;
	}

	//Region 1.5
	else if(bounds.inf() >= -1.573 && bounds.inf() <= -1.571){

	        Real bound = Real(0.61009);

		//sup is in Region 2
	        if(bounds.sup() <= -0.422){
		        Real rbound = tanhDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is beyond global max
		else if(bounds.sup() >= -0.422){
		        bound = Real(4.0859);
		}
		  
	        return bound;
	}

	//Region 1
	else if(bounds.inf() <= -1.573){

	        Real bound = tanhDer(4, Real(bounds.inf())).abs();

		//sup is in Region 1
	        if(bounds.sup() <= -1.573){
		        Real rbound = tanhDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is in Region 2
	        if(bounds.sup() <= -0.422){

		        bound = Real(0.61009);
			  
		        Real rbound = tanhDer(4, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}		

		//sup is beyond global max
		else if(bounds.sup() >= -0.422){
		        bound = Real(4.0859);
		}
		  
	        return bound;
	}	
  
        return Real(4.0859);
}

//The hardcoded bounds used in this function are conservative numerical bounds
Real getTanh3rdDerBound(Interval bounds){

        //Region 4
        if(bounds.inf() >= 1.147){
	        return tanhDer(3, Real(bounds.inf()));
	}

	//Region 3.5
	else if(bounds.inf() >= 1.145 && bounds.inf() <= 1.147){
	        return Real(0.66667);
	}

	//Region 3
        else if(bounds.inf() >= 0 && bounds.inf() <= 1.145){

	        Real bound = tanhDer(3, Real(bounds.inf())).abs();

		//sup is in Region 3
	        if(bounds.sup() <= 1.145){
		  
		        Real rbound = tanhDer(3, Real(bounds.sup())).abs();
			
			if (rbound > bound) bound = rbound;
		}		
		//sup is in Region 4
		else{
		        if (Real(0.66667) > bound) bound = Real(0.66667);
		}
		
	        return bound;
	}

	//Region 2
        else if(bounds.inf() >= -1.145 && bounds.inf() <= 0){
	  
	        Real bound = tanhDer(3, Real(bounds.inf())).abs();

		//sup is in Region 2
	        if(bounds.sup() <= 0){
		        Real rbound = tanhDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		else{
		        bound = Real(2);
		}
		
	        return bound;
	}

	//Region 1.5
        else if(bounds.inf() >= -1.147 && bounds.inf() <= -1.145){
	  
	        Real bound = Real(0.66667);

		//sup is in Region 2
	        if(bounds.sup() <= 0){
		        Real rbound = tanhDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		else{
		        bound = Real(2);
		}
		
	        return bound;
	}

	//Region 1
	else if(bounds.inf() <= -1.147){

	        Real bound = tanhDer(3, Real(bounds.inf())).abs();

		//sup is in Region 1
	        if(bounds.sup() <= -1.147){
		        Real rbound = tanhDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is in Region 2
		else if(bounds.sup() >= -1.147 && bounds.sup() <= 0){
		        bound = Real(0.66667);

			Real rbound = tanhDer(3, Real(bounds.sup())).abs();

			if (rbound > bound) bound = rbound;
		}

		//sup is beyond global max
		else if(bounds.sup() >= 0){
		        bound = Real(2);
		}
		  
	        return bound;
	}		
  
        return Real(2);
}

Real getSwish4thDerBound(Interval bounds){

        if(bounds.inf() <= 2.2 && bounds.sup() >= -2.2){
	        return Real(0.13);
	}

        if(bounds.inf() > 2.2 && bounds.inf() <= 6){
	        return Real(0.04);
	}
	
        if(bounds.sup() >= -6 && bounds.sup() < -2.2){
	        return Real(0.04);
	}

	if(bounds.inf() > 6){
	        return Real(0.013);
	}

	if(bounds.sup() < -6){
	        return Real(0.013);
	}		
  
        return Real(0.13);
}

Real getSwish3rdDerBound(Interval bounds){

        if(bounds.inf() <= 3 && bounds.sup() >= -3){
	        return Real(0.31);
	}

        if(bounds.inf() > 3 && bounds.inf() <= 5){
	        return Real(0.025);
	}

        if(bounds.sup() >= -5 && bounds.sup() < -3){
	        return Real(0.025);
	}			

	if(bounds.inf() > 5){
	        return Real(0.013);
	}

	if(bounds.sup() < -5){
	        return Real(0.013);
	}		
  
        return Real(0.31);
}

Real getSwishTen4thDerBound(Interval bounds){

        if(bounds.inf() <= 0.071 && bounds.sup() >= -0.071){
	        return Real(500);
	}

        if(bounds.inf() > 0.071 && bounds.inf() <= 0.4){
	        return Real(204);
	}

	if(bounds.inf() > 0.4 && bounds.inf() <= 1.2){
	        return Real(10);
	}

        if(bounds.sup() >= -0.4 && bounds.sup() < -0.071){
	        return Real(204);
	}

	if(bounds.sup() >= -1.2 && bounds.sup() < -0.4){
	        return Real(10);
	}	

	if(bounds.inf() > 1.2){
	        return Real(0.05);
	}

	if(bounds.sup() < -1.2){
	        return Real(0.05);
	}		
  
        return Real(500);
}

Real getSwishTen3rdDerBound(Interval bounds){

        if(bounds.inf() <= 0.3 && bounds.sup() >= -0.3){
	        return Real(30.9);
	}

        if(bounds.inf() > 0.3 && bounds.inf() <= 0.91){
	        return Real(2.6);
	}

        if(bounds.sup() >= -0.91 && bounds.sup() < -0.3){
	        return Real(2.6);
	}			

	if(bounds.inf() > 0.91){
	        return Real(0.07);
	}

	if(bounds.sup() < -0.91){
	        return Real(0.07);
	}		
  
        return Real(30.9);
}

Real getSwishHundred4thDerBound(Interval bounds){

        if(bounds.inf() <= 0.007 && bounds.sup() >= -0.007){
	        return Real(500000);
	}

        if(bounds.inf() > 0.007 && bounds.inf() <= 0.045){
	        return Real(200000);
	}

        if(bounds.inf() >= -0.045 && bounds.inf() <= -0.007){
	        return Real(200000);
	}

	if(bounds.inf() > 0.045 && bounds.inf() <= 0.12){
	        return Real(4700);
	}

        if(bounds.sup() >= -0.12 && bounds.sup() < -0.045){
	        return Real(4700);
	}

	if(bounds.sup() >= 0.12 && bounds.sup() < 0.17){
	        return Real(50);
	}

	if(bounds.sup() >= -0.17 && bounds.sup() < -0.12){
	        return Real(50);
	}	

	if(bounds.inf() > 0.17){
	        return Real(0.55);
	}

	if(bounds.sup() < -0.17){
	        return Real(0.55);
	}		
  
        return Real(500000);
}

Real getSwishHundred3rdDerBound(Interval bounds){

        if(bounds.inf() <= 0.2 && bounds.sup() >= -0.2){
	        return Real(7400);
	}

        if(bounds.inf() > 0.2 && bounds.inf() <= 0.5){
	        return Real(8800);
	}

        if(bounds.sup() >= -0.5 && bounds.sup() < -0.2){
	        return Real(8800);
	}

        if(bounds.inf() > 0.5 && bounds.inf() <= 0.8){
	        return Real(3000);
	}

        if(bounds.sup() >= -0.8 && bounds.sup() < -0.5){
	        return Real(3000);
	}

        if(bounds.inf() > 0.8 && bounds.inf() <= 1.2){
	        return Real(260);
	}

        if(bounds.sup() >= -1.2 && bounds.sup() < -0.8){
	        return Real(260);
	}	

	if(bounds.inf() > 1.2){
	        return Real(7.5);
	}

	if(bounds.sup() < -1.2){
	        return Real(7.5);
	}		
  
        return Real(8800);
}

void dnn::sig_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars){

    Real midPoint = Real(intC.midpoint());
    Real apprPoint = sigmoid(midPoint);

    //NB: This assumes a 2nd order TS approximation
    Real coef1 = sigDer(1, midPoint);
    Real coef2 = sigDer(2, midPoint)/2;
    Real coef3 = sigDer(3, midPoint)/6;						

    Real derBound = getSig3rdDerBound(intC);

    Real maxDev = Real(intC.sup()) - midPoint;

    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }
    
    Real fact = Real(6);					       
    maxDev.pow_assign(3);
    Real remainder = (derBound * maxDev) / fact;

    Interval apprInt = Interval(apprPoint);

    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInd + 1] = 3;
    
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));
    
    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
    
    Polynomial exp = deg0Poly + deg1Poly + deg2Poly;
    Interval rem;
    remainder.to_sym_int(rem);
    
    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);
	derBound = getSig4thDerBound(intC);
      
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);
	
	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint),
						  numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }

    tmReset.expansion = exp;
    tmReset.remainder = rem;
}

void dnn::swish_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars){

    Real midPoint = Real(intC.midpoint());  
    Real apprPoint = swish(midPoint);
						
    //First try a 2nd order TS approximation
    
    Real coef1 = swish1stDer(midPoint);
    Real coef2 = swish2ndDer(midPoint)/2;
    Real coef3 = swish3rdDer(midPoint)/6;
						

    Real derBound = getSwish3rdDerBound(intC);
						
    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }
						
    Real fact = Real(6);					       
    maxDev.pow_assign(3);
    Real remainder = (derBound * maxDev) / fact;

    Interval apprInt = Interval(apprPoint);

    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInd + 1] = 3;
						
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));
						
    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
						
    Polynomial exp = deg0Poly + deg1Poly + deg2Poly;
    Interval rem;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);

	derBound = getSwish4thDerBound(intC);
	
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);

	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) + Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }

    tmReset.expansion = exp;
    tmReset.remainder = rem;
    
}

void dnn::swish10_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars){

    Real midPoint = Real(intC.midpoint());  
    Real apprPoint = swishTen(midPoint);
						
    //First try a 2nd order TS approximation
    
    Real coef1 = swishTen1stDer(midPoint);
    Real coef2 = swishTen2ndDer(midPoint)/2;
    Real coef3 = swishTen3rdDer(midPoint)/6;
						

    Real derBound = getSwishTen3rdDerBound(intC);
						
    Real maxDev = Real(intC.sup()) - midPoint;
    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }
						
    Real fact = Real(6);					       
    maxDev.pow_assign(3);
    Real remainder = (derBound * maxDev) / fact;

    Interval apprInt = Interval(apprPoint);

    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInd + 1] = 3;
						
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));
						
    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
						
    Polynomial exp = deg0Poly + deg1Poly + deg2Poly;
    Interval rem;
    remainder.to_sym_int(rem);

    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);

	derBound = getSwishTen4thDerBound(intC);
	
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);

	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) + Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }

    tmReset.expansion = exp;
    tmReset.remainder = rem;
    
}

void dnn::tanh_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars){

    Real midPoint = Real(intC.midpoint());
    Real apprPoint = tanh(midPoint);

    //NB: This assumes a 2nd order TS approximation
    Real coef1 = tanhDer(1, midPoint);
    Real coef2 = tanhDer(2, midPoint)/2;
    Real coef3 = tanhDer(3, midPoint)/6;						

    Real derBound = getTanh3rdDerBound(intC);

    Real maxDev = Real(intC.sup()) - midPoint;

    if (midPoint - Real(intC.inf()) > maxDev){
        maxDev = midPoint - Real(intC.inf());
    }
    
    Real fact = Real(6);					       
    maxDev.pow_assign(3);
    Real remainder = (derBound * maxDev) / fact;

    Interval apprInt = Interval(apprPoint);

    Interval deg1Int = Interval(coef1);
    Interval deg2Int = Interval(coef2);
    Interval deg3Int = Interval(coef3);

    std::vector<int> deg1(numVars, 0);
    deg1[varInd + 1] = 1;
    std::vector<int> deg2(numVars, 0);
    deg2[varInd + 1] = 2;
    std::vector<int> deg3(numVars, 0);
    deg3[varInd + 1] = 3;
    
    Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));
    
    Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
      Polynomial(Monomial(deg1Int, deg1));

    Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
      Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
      Polynomial(Monomial(deg2Int, deg2));
    
    Polynomial exp = deg0Poly + deg1Poly + deg2Poly;
    Interval rem;
    remainder.to_sym_int(rem);
    
    //if uncertainty too large, use a 3rd order approximation
    if (rem.width() > 0.00001){
        fact = 24;
	maxDev = Real(intC.sup()) - midPoint;
	maxDev.pow_assign(4);
	derBound = getTanh4thDerBound(intC);
      
	remainder = (derBound * maxDev) / fact;
	remainder.to_sym_int(rem);
	
	Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint),
						  numVars)) +
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	  Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	  Polynomial(Monomial(deg3Int, deg3));
						
	exp += deg3Poly;
    }

    tmReset.expansion = exp;
    tmReset.remainder = rem;
}

void dnn::relu_reset(TaylorModel &tmReset, const Interval intC, const int varInd, const int numVars){


    Polynomial exp;
    Interval rem;

    //ReLU all in 0 area
    if(intC.sup() < 0){

        Interval zeroInt = Interval(0.0, 0.0);
						
	exp = Polynomial(Monomial(zeroInt, numVars));
	rem = zeroInt;
    }
    
    //ReLU all in positive area
    else if(intC.inf() > 0){

        std::vector<int> deg1(numVars, 0);
	deg1[varInd + 1] = 1;

	Polynomial deg1Poly = Polynomial(Monomial(Interval(1.0, 1.0), deg1));

	exp = deg1Poly;
	rem = Interval(0.0, 0.0);

    }					

    //ReLU in both areas: approximate using the Swish function
    else{
        Real midPoint = Real(intC.midpoint());
	
        Interval apprPoint = swishHundred(midPoint);
	
	//NB: This assumes a 2nd order TS approximation

	Real coef1 = swishHundred1stDer(midPoint);
	Real coef2 = swishHundred2ndDer(midPoint)/2;
	Real coef3 = swishHundred3rdDer(midPoint)/6;
						
	Real derBound = getSwishHundred3rdDerBound(intC);

	Real maxDev = Real(intC.sup()) - midPoint;
	if (midPoint - Real(intC.inf()) > maxDev){
	    maxDev = midPoint - Real(intC.inf());
	}
						
	Real fact = Real(6);					       
	maxDev.pow_assign(3);
	Real remainder = (derBound * maxDev) / fact;

	Interval apprInt = Interval(apprPoint);

	Interval deg1Int = Interval(coef1);
	Interval deg2Int = Interval(coef2);
	Interval deg3Int = Interval(coef3);
	
	std::vector<int> deg1(numVars, 0);
	deg1[varInd + 1] = 1;
	std::vector<int> deg2(numVars, 0);
	deg2[varInd + 1] = 2;
	std::vector<int> deg3(numVars, 0);
	deg3[varInd + 1] = 3;
						
	Polynomial deg0Poly = Polynomial(Monomial(apprInt, numVars));
	
	Polynomial deg1Poly = Polynomial(Monomial(Interval(Real(-1) * coef1 * midPoint), numVars)) +
	  Polynomial(Monomial(deg1Int, deg1));
	
	Polynomial deg2Poly = Polynomial(Monomial(Interval(coef2 * midPoint * midPoint), numVars)) -
	  Polynomial(Monomial(Interval(Real(2) * coef2 * midPoint), deg1)) +
	  Polynomial(Monomial(deg2Int, deg2));
	
	exp = deg0Poly + deg1Poly + deg2Poly;
	remainder.to_sym_int(rem);

	//if uncertainty too large, use a 3rd order approximation
	if (rem.width() > 0.00001){
	    fact = 24;
	    maxDev = Real(intC.sup()) - midPoint;
	    maxDev.pow_assign(4);
	    
	    derBound = getSwishHundred4thDerBound(intC);
	    
	    remainder = (derBound * maxDev) / fact;
	    remainder.to_sym_int(rem);
	    
	    Polynomial deg3Poly = Polynomial(Monomial(Interval(Real(-1) * coef3 * midPoint * midPoint * midPoint), numVars)) +
	      Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint * midPoint), deg1)) -
	      Polynomial(Monomial(Interval(Real(3) * coef3 * midPoint), deg2)) +
	      Polynomial(Monomial(deg3Int, deg3));

	    
	    exp += deg3Poly;
	}
	
	//Add approximation error between ReLU and Swish
	// exp += Polynomial(Monomial(Interval(0.0014, 0.0014), numVars));
	// rem += Interval(-0.0014, 0.0014);
	
	//Add approximation error between ReLU and Swish
	
	if (intC.inf() > -0.0127){
	    double maxDer = -swishHundred(Real(intC.inf())).getValue_RNDD();
	  
	    if(intC.sup() - swishHundred(Real(intC.sup())).getValue_RNDD() > maxDer){
	        maxDer = intC.sup() - swishHundred(Real(intC.sup())).getValue_RNDD();
	    }
	    
	    rem += Interval(0.0, maxDer);
	}
	else{						  
	    rem += Interval(0.0, 0.0028);
	}
	printf("input interval: [%f, %f]\n", intC.inf(), intC.sup());
	printf("returned remainder: [%f, %f]\n", rem.inf(), rem.sup());
    }

    tmReset.expansion = exp;
    tmReset.remainder = rem;    
  
}

void dnn::load_dnn(std::vector<ResetMap> &resets, std::vector<dnn::activation> &activations, const Variables &vars, const std::string filename) {
    Node dnn = LoadFile(filename);

    Node weights = dnn["weights"];
    Node offsets = dnn["offsets"];
    Node acts = dnn["activations"];

    for(int i=0; i<weights.size(); i++) {
  
	int layerId = i+1;
	std::string activationFcn = acts[layerId].as<string>();
	dnn::activation activationAsEnum = LINEAR;

	if(!strncmp(activationFcn.c_str(), "Tanh", strlen("Tanh"))){
	        activationAsEnum = dnn::TANH;
	}
	else if(!strncmp(activationFcn.c_str(), "Sigmoid", strlen("Sigmoid"))){
	        activationAsEnum = dnn::SIGMOID;
	}
	else if(!strncmp(activationFcn.c_str(), "Swish", strlen("Swish"))){
	        activationAsEnum = dnn::SWISH;
	}	
	else if(!strncmp(activationFcn.c_str(), "Relu", strlen("Relu"))){
	        activationAsEnum = dnn::RELU;
	}
	

        map<string,TaylorModel> taylorModels;
        Node layer = weights[layerId];
        int layerSize = layer[0].size();

        int currentNeuron = 1;
        for(const_iterator neuronIt=layer.begin(); neuronIt != layer.end(); ++neuronIt) {
            int currentWeight = 1;
            stringstream buffer;
            for(int i = 0; i < layerSize; i++) {

                buffer << (*neuronIt)[i];
                buffer << " * " << "f" << currentWeight << + " + ";

                currentWeight++;
            }

            buffer << offsets[layerId][currentNeuron-1];

            taylorModels["f" + to_string(currentNeuron)] = TaylorModel(buffer.str(), vars);            

            currentNeuron++;
        }

        TaylorModelVec tms;
	std::vector<bool> isIdentity(vars.size() - 1);

        for(int i=0; i < vars.size() - 1; i++ ){
            string varName = vars.varNames[i+1];

            if(taylorModels.find(varName) == taylorModels.end()) {

	        if(varName[0] == 'f'){
		    tms.tms.push_back(TaylorModel("0", vars));
		    isIdentity[i] = false;
		}
		else{
		    tms.tms.push_back(TaylorModel(varName, vars));
		    isIdentity[i] = true;
		}
            } else {
                tms.tms.push_back(taylorModels[varName]);
                isIdentity[i] = false;
            }
        }

        resets.push_back(ResetMap(tms, isIdentity));
        activations.push_back(activationAsEnum);
	
    }
}
