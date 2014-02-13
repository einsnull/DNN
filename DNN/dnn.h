#include "SAE.h"
#include "FunctionBase.h"
#include "SoftMax.h"
#include "getConfig.h"

class DNN : public FunctionBase
{
private:
	MatrixXd saeTheta1;
	MatrixXd saeTheta2;
	MatrixXd saeB1;
	MatrixXd saeB2;
	MatrixXd softMaxTheta;
	int numClasses;
	int sae1HiddenSize;
	int sae2HiddenSize;
	int inputSize;
public:
	DNN(int sae1HiddenSize,int sae2HiddenSize,int numClasses);
	MatrixXi predict(
		MatrixXd &data);
	void fineTune(MatrixXd &data,
		MatrixXi &labels,double lambda,
		double alpha,int maxIter,int batchSize);
	void preTrain(MatrixXd &data,MatrixXi &labels,
		double lambda[],double alpha[],double beta[],double sparsityParam[],
		int maxIter[],int miniBatchSize,int imgWidth);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
private:
	MatrixXd softmaxGradient(MatrixXd &x);
	MatrixXd feedForward(MatrixXd &theta,
		MatrixXd &b,MatrixXd data);
	void updateParameters(MatrixXd &theta1Grad,MatrixXd &theta2Grad,
						   MatrixXd &b1Grad,MatrixXd &b2Grad,
						   MatrixXd &softmaxTheta,double alpha);
	double computeCost(MatrixXd &theta1Grad,
		MatrixXd &b1Grad,MatrixXd &theta2Grad,MatrixXd &b2Grad,
		MatrixXd &softmaxThetaGrad,MatrixXd &data,
		MatrixXi &labels,double lambda);
};

DNN::DNN(int sae1HiddenSize,int sae2HiddenSize,int numClasses)
{
	this->sae1HiddenSize = sae1HiddenSize;
	this->sae2HiddenSize = sae2HiddenSize;
	this->numClasses = numClasses;
}

MatrixXd DNN::feedForward(MatrixXd &theta,MatrixXd &b,
									 MatrixXd data)
{
	int m = data.cols();
	MatrixXd z2 = theta * data + b.replicate(1,m);
	MatrixXd a2 = sigmoid(z2);
	return a2;
}


MatrixXi DNN::predict(
		MatrixXd &data)
{
	MatrixXd term1 = saeTheta1 * data;
	MatrixXd z2 = bsxfunPlus(term1,saeB1);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd term2 = saeTheta2 * a2;
	MatrixXd z3 = bsxfunPlus(term2,saeB2);
	MatrixXd a3 = sigmoid(z3);
	MatrixXd z4 = softMaxTheta * a3;
	MatrixXd a4 = expMat(z4);
	MatrixXd a4ColSum = a4.colwise().sum();
	a4 = bsxfunRDivide(a4,a4ColSum);
	MatrixXi pred(1,a4.cols());
	for(int i = 0;i < a4.cols();i++)
	{
		double max = 0;
		int idx = 0;
		for(int j = 0;j < a4.rows();j++)
		{
			if(a4(j,i) > max)
			{
				idx = j;
				max = a4(j,i);
			}
		}
		pred(0,i) = idx;
	}
	return pred;
}

//component wise softmax gradient
MatrixXd DNN::softmaxGradient(MatrixXd &x)
{
	MatrixXd negX = x * (-1);
	MatrixXd expX = expMat(negX);
	MatrixXd term1 = (MatrixXd::Ones(expX.rows(),expX.cols())
		+ expX).array().square();
	MatrixXd grad = expX.cwiseQuotient(term1);
	return grad;
}


void DNN::updateParameters(MatrixXd &theta1Grad,MatrixXd &theta2Grad,
						   MatrixXd &b1Grad,MatrixXd &b2Grad,
						   MatrixXd &softmaxThetaGrad,double alpha)
{
	saeTheta1 -= theta1Grad * alpha;
	saeTheta2 -= theta2Grad * alpha;
	saeB1 -= b1Grad * alpha;
	saeB2 -= b2Grad * alpha;
	softMaxTheta -= softmaxThetaGrad * alpha;
}

double DNN::computeCost(MatrixXd &theta1Grad,
		MatrixXd &b1Grad,MatrixXd &theta2Grad,
		MatrixXd &b2Grad,MatrixXd &softmaxThetaGrad,
		MatrixXd &data,MatrixXi &labels,double lambda)
{
	MatrixXd groundTruth = binaryCols(labels,numClasses);
	int M = labels.rows();
	//forward calculate
	MatrixXd term1 = saeTheta1 * data;
	MatrixXd z2 = bsxfunPlus(term1,saeB1);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd term2 = saeTheta2 * a2;
	MatrixXd z3 = bsxfunPlus(term2,saeB2);
	MatrixXd a3 = sigmoid(z3);
	MatrixXd z4 = softMaxTheta * a3;
	MatrixXd a4 = expMat(z4);
	MatrixXd a4ColSum = a4.colwise().sum();
	a4 = bsxfunRDivide(a4,a4ColSum);

	MatrixXd delta4 = a4 - groundTruth;
	MatrixXd delta3 = (softMaxTheta.transpose() * delta4).cwiseProduct(sigmoidGradient(z3));
	MatrixXd delta2 = (saeTheta2.transpose() * delta3).cwiseProduct(sigmoidGradient(z2));

	softmaxThetaGrad = (groundTruth - a4) * a3.transpose() * (-1.0 / M) + softMaxTheta * lambda;

	theta2Grad = delta3 * a2.transpose() * (1.0 / M) + saeTheta2 * lambda;
	b2Grad = delta3.rowwise().sum() * (1.0 / M);
	theta1Grad = delta2 * data.transpose() * (1.0 / M) + saeTheta1 * lambda;
	b1Grad = delta2.rowwise().sum() * (1.0 / M);

	double cost = (-1.0 / M) * (groundTruth.cwiseProduct(logMat(a4))).array().sum()
		+ lambda / 2.0 * softMaxTheta.array().square().sum()
		+ lambda / 2.0 * saeTheta1.array().square().sum()
		+ lambda / 2.0 * saeTheta2.array().square().sum();

	return cost;
}

void DNN::fineTune(MatrixXd &data,MatrixXi &labels,
				   double lambda,double alpha,int maxIter,int batchSize)
{
	MatrixXd theta1Grad(saeTheta1.rows(),saeTheta1.cols());
	MatrixXd theta2Grad(saeTheta2.rows(),saeTheta2.cols());
	MatrixXd b1Grad(saeB1.rows(),saeB1.cols());
	MatrixXd b2Grad(saeB2.rows(),saeB2.cols());
	MatrixXd softmaxThetaGrad(softMaxTheta.rows(),softMaxTheta.cols());
	MatrixXd miniTrainData(data.rows(),batchSize);
	MatrixXi miniLabels(batchSize,1);
	int iter = 1;
	int numBatches = data.cols() / batchSize;
	
	//mini batch stochastic gradient decent
	for(int i = 0; i < maxIter;i++)
	{
		double J = 0;
		// compute the cost
		for(int j = 0;j < numBatches; j++)
		{
			miniTrainData = data.middleCols(j * batchSize,batchSize);
			miniLabels = labels.middleRows(j * batchSize,batchSize);
			J += computeCost(theta1Grad,b1Grad,theta2Grad,
				b2Grad,softmaxThetaGrad,miniTrainData,miniLabels,lambda);
#ifdef _IOSTREAM_
			if(miniTrainData.cols() < 1 || miniTrainData.rows() < 1)
			{
				cout << "Too few training examples!"  << endl; 
			}
#endif

			if(fabs(J) < 0.001)
			{
				break;
			}
			updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,softmaxThetaGrad,alpha);
		}
		J = J / numBatches;
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
	}
}

void DNN::preTrain(MatrixXd &data,MatrixXi &labels,
		double lambda[],double alpha[],double beta[],double sparsityParam[],
		int maxIter[],int miniBatchSize,int imgWidth)
{
	int numOfExamples = data.cols();
	int ndim = data.rows(); 
	inputSize = ndim;
	//cout << "ndim : " << ndim << endl;
	SAE sae1(ndim,sae1HiddenSize);
	cout << "PreTraining sae1 ..." << endl;
	sae1.train(data,lambda[0],alpha[0],beta[0],
		sparsityParam[0],maxIter[0],MINI_BATCH_SGD,&miniBatchSize);
	
	MatrixXd theta1 = sae1.getTheta();
	saeTheta1.resize(theta1.rows(),theta1.cols());
	saeTheta1 = theta1;
	MatrixXd b1 = sae1.getBias();
	saeB1.resize(b1.rows(),b1.cols());
	saeB1 = b1;

	/*cout << "saeTheta1:" << endl;
	cout << saeTheta1.rows() << " " << saeTheta1.cols() << endl;*/
	
	buildImage(theta1,imgWidth,"sae1.jpg",false);
	
	MatrixXd sae1Features = feedForward(saeTheta1,saeB1,data);
	SAE sae2(sae1HiddenSize,sae2HiddenSize);
	cout << "PreTraining sae2 ..." << endl;
	sae2.train(sae1Features,lambda[1],alpha[1],beta[1],
		sparsityParam[1],maxIter[1],MINI_BATCH_SGD,&miniBatchSize);

	MatrixXd theta2 = sae2.getTheta();
	saeTheta2.resize(theta2.rows(),theta2.cols());
	saeTheta2 = theta2;
	MatrixXd b2 = sae2.getBias();
	saeB2.resize(b2.rows(),b2.cols());
	saeB2 = b2;
	
	/*cout << "saeTheta2:" << endl;
	cout << saeTheta2.rows() << " " << saeTheta2.cols() << endl;*/
	MatrixXd filter = saeTheta2 * saeTheta1;
	buildImage(filter,imgWidth,"sae2.jpg",false);

	MatrixXd sae2Features = feedForward(saeTheta2,saeB2,sae1Features);
	//cout << "Saving sae2Features..." << endl;
	//saveMatrix(sae2Features,"saeFeatures2");
	cout << "PreTraining softmax ..." << endl;
	SoftMax softmax(sae2HiddenSize,numClasses);
	softmax.train(sae2Features,labels,lambda[2],alpha[2],maxIter[2],miniBatchSize);
	MatrixXd smTheta = softmax.getTheta();
	softMaxTheta.resize(smTheta.rows(),smTheta.cols());
	softMaxTheta = smTheta;
}

//save model to file
bool DNN::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	ofs << inputSize << " " << sae1HiddenSize << " "
		<< sae2HiddenSize << " " << numClasses << endl;
	for(i = 0; i < saeTheta1.rows(); i++)
	{
		for(j = 0;j < saeTheta1.cols(); j++)
		{
			ofs << saeTheta1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < saeTheta2.rows(); i++)
	{
		for(j = 0;j < saeTheta2.cols(); j++)
		{
			ofs << saeTheta2(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < saeB1.rows(); i++)
	{
		for(j = 0; j < saeB1.cols(); j++) 
		{
			ofs << saeB1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < saeB2.rows(); i++)
	{
		for(j = 0; j < saeB2.cols(); j++) 
		{
			ofs << saeB2(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < softMaxTheta.rows(); i++)
	{
		for(j = 0; j < softMaxTheta.cols(); j++)
		{
			ofs << softMaxTheta(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

//load model from file
bool DNN::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	
	ifs >> inputSize >> sae1HiddenSize >> sae2HiddenSize >> numClasses;
	int i,j;
	saeTheta1.resize(sae1HiddenSize,inputSize);
	saeTheta2.resize(sae2HiddenSize,sae1HiddenSize);
	saeB1.resize(sae1HiddenSize,1);
	saeB2.resize(sae2HiddenSize,1);

	for(i = 0; i < saeTheta1.rows(); i++)
	{
		for(j = 0;j < saeTheta1.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> saeTheta1(i,j);
		}
	}
	for(i = 0; i < saeTheta2.rows(); i++)
	{
		for(j = 0;j < saeTheta2.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> saeTheta2(i,j);
		}
	}
	for(i = 0; i < saeB1.rows(); i++)
	{
		for(j = 0; j < saeB1.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> saeB1(i,j);
		}
	}
	for(i = 0; i < saeB2.rows(); i++)
	{
		for(j = 0; j < saeB2.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> saeB2(i,j);
		}
	}
	for(i = 0; i < softMaxTheta.rows(); i++)
	{
		for(j = 0; j < softMaxTheta.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> softMaxTheta(i,j);
		}
	}
	ifs.close();
	return true;
}