#include <iostream>
#include "dataAndImage.h"
#include "dnn.h"
using namespace std;

int main()
{
	int sae1HiddenSize = 100;
	int sae2HiddenSize = 100;
	int numClasses = 10;
	int imgWidth = 28;
	double lambda[4] = {3e-3,3e-3,3e-3,1e-4};
	double alpha[4] = {0.2,0.2,0.2,0.2};
	double beta[2] = {3,3};
	double sparsityParam[2] = {0.1,0.1};
	int maxIter[4] = {100,100,100,200};
	int miniBatchSize = 1000;
	MatrixXd trainingData(1,1);
	MatrixXi trainingLabels(1,1);
	MatrixXd testData(1,1);
	MatrixXi testLabels(1,1);
	char *fileBuf  = new char[4096];
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"lambda0:",lambda[0]);
		getConfigDoubleValue(fileBuf,"lambda1:",lambda[1]);
		getConfigDoubleValue(fileBuf,"lambda2:",lambda[2]);
		getConfigDoubleValue(fileBuf,"lambda3:",lambda[3]);
		getConfigDoubleValue(fileBuf,"alpha0:",alpha[0]);
		getConfigDoubleValue(fileBuf,"alpha1:",alpha[1]);
		getConfigDoubleValue(fileBuf,"alpha2:",alpha[2]);
		getConfigDoubleValue(fileBuf,"alpha3:",alpha[3]);
		getConfigDoubleValue(fileBuf,"beta0:",beta[0]);
		getConfigDoubleValue(fileBuf,"beta1:",beta[1]);
		getConfigDoubleValue(fileBuf,"sparseParam0:",sparsityParam[0]);
		getConfigDoubleValue(fileBuf,"sparseParam1:",sparsityParam[1]);
		getConfigIntValue(fileBuf,"maxIter0:",maxIter[0]);
		getConfigIntValue(fileBuf,"maxIter1:",maxIter[1]);
		getConfigIntValue(fileBuf,"maxIter2:",maxIter[2]);
		getConfigIntValue(fileBuf,"maxIter3:",maxIter[3]);
		getConfigIntValue(fileBuf,"miniBatchSize:",miniBatchSize);
		getConfigIntValue(fileBuf,"sae1HiddenSize:",sae1HiddenSize);
		getConfigIntValue(fileBuf,"sae2HiddenSize:",sae2HiddenSize);
		getConfigIntValue(fileBuf,"imgWidth:",imgWidth);
		cout << "lambda0: " << lambda[0] << endl;
		cout << "lambda1: " << lambda[1] << endl;
		cout << "lambda2: " << lambda[2] << endl;
		cout << "lambda3: " << lambda[3] << endl;
		cout << "alpha0: " << alpha[0] << endl;
		cout << "alpha1: " << alpha[1] << endl;
		cout << "alpha2: " << alpha[2] << endl;
		cout << "alpha3: " << alpha[3] << endl;
		cout << "beta0: " << beta[0] << endl;
		cout << "beta1: " << beta[1] << endl;
		cout << "sparseParam0: " << sparsityParam[0] << endl;
		cout << "sparseParam1: " << sparsityParam[1] << endl;
		cout << "maxIter0: " << maxIter[0] << endl;
		cout << "maxIter1: " << maxIter[1] << endl;
		cout << "maxIter2: " << maxIter[2] << endl;
		cout << "maxIter3: " << maxIter[3] << endl;
		cout << "miniBatchSize: " << miniBatchSize << endl;
		cout << "sae1HiddenSize: " << sae1HiddenSize << endl;
		cout << "sae2HiddenSize: " << sae2HiddenSize << endl;
		cout << "imgWidth: " << imgWidth << endl;
	}
	delete []fileBuf;
	//timer
	clock_t start = clock();
	ret = loadMnistData(trainingData,"mnist\\train-images-idx3-ubyte");
	cout << "Loading training data..." << endl;
	if(ret == false)
	{
		return -1;
	}
	ret = loadMnistLabels(trainingLabels,"mnist\\train-labels-idx1-ubyte");
	if(ret == false)
	{
		return -1;
	}
	MatrixXd showData = trainingData.leftCols(100).transpose();
	buildImage(showData,imgWidth,"data.jpg",false);

	DNN dnn(sae1HiddenSize,sae2HiddenSize,numClasses);
	dnn.preTrain(trainingData,trainingLabels,lambda,alpha,beta,
		sparsityParam,maxIter,miniBatchSize,imgWidth);
	cout << "Loading test data..." << endl;
	ret = loadMnistData(testData,"mnist\\t10k-images-idx3-ubyte");
	if(ret == false)
	{
		return -1;
	}
	ret = loadMnistLabels(testLabels,"mnist\\t10k-labels-idx1-ubyte");
	if(ret == false)
	{
		return -1;
	}
	MatrixXi pred1 = dnn.predict(testData);
	double acc1 = dnn.calcAccurancy(testLabels,pred1);
	cout << "Accurancy before fine tuning: " << acc1 * 100 << "%" << endl;
	cout << "Fine Tuning..." << endl;
	dnn.fineTune(trainingData,trainingLabels,lambda[3],
		alpha[3],maxIter[3],miniBatchSize);
	MatrixXi pred2 = dnn.predict(testData);
	double acc2 = dnn.calcAccurancy(testLabels,pred2);
	cout << "Accurancy: " << acc2 * 100 << "%" << endl;
	dnn.saveModel("DNN_Model.txt");
	clock_t end = clock();
	cout << "The code ran for " << 
		(end - start)/(double)(CLOCKS_PER_SEC*60) << " minutes." << endl;
	system("pause");
	return 0;
}

