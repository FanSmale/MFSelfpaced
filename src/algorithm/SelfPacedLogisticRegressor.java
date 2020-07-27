package algorithm;

import java.io.FileReader;
import java.util.Arrays;

import Jama.Matrix;
import common.SimpleTools;
import weka.core.Instances;

/**
 * Self-paced logistic regression. <br>
 * Project: Self-paced learning.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/MFAdaBoosting.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 *         Data Created: July 26, 2020.<br>
 *         Last modified: July 26, 2020.
 * @version 1.0
 */
public class SelfPacedLogisticRegressor {

	/**
	 * The whole input data. The first column is always 1.
	 */
	double[][] wholeX;

	/**
	 * The whole output data.
	 */
	double[][] wholeY;

	/**
	 * The training data input.
	 */
	double[][] trainingX;

	/**
	 * The training data output.
	 */
	double[][] trainingY;

	/**
	 * The testing data input.
	 */
	double[][] testingX;

	/**
	 * The testing data output.
	 */
	double[][] testingY;

	/**
	 * The weights for the training hyper-space.
	 */
	double[] weights;

	/**
	 * The initial distance threshold.
	 */
	double distanceThresholdInitial = 2.0;

	/**
	 * The decrement distance threshold.
	 */
	double distanceThresholdDecrement = 0.1;

	/**
	 * Maximal change loops for adjusting the distance threshold.
	 */
	int maxChangeLoops = 100;

	/**
	 * Instance proportion threshold.
	 */
	double instanceProportionThreshold = 0.9;

	/**
	 * Is the training data removed?.
	 */
	boolean[] removalArray;

	/**
	 ****************** 
	 * The first constructor.
	 * 
	 * @param paraTrainingFilename
	 *            The data filename.
	 ****************** 
	 */
	public SelfPacedLogisticRegressor(String paraTrainingFilename) {
		Instances data = null;
		// Step 1. Read training set.
		try {
			FileReader tempFileReader = new FileReader(paraTrainingFilename);
			data = new Instances(tempFileReader);
			tempFileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraTrainingFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		int tempNumInstances = data.numInstances();
		int tempNumAttributes = data.numAttributes();

		wholeX = new double[tempNumInstances][tempNumAttributes];
		wholeY = new double[tempNumInstances][1];

		for (int i = 0; i < tempNumInstances; i++) {
			// The first element is always set to 1.
			wholeX[i][0] = 1;
			wholeY[i][0] = data.instance(i).value(tempNumAttributes - 1);
			for (int j = 0; j < tempNumAttributes - 1; j++) {
				wholeX[i][j + 1] = data.instance(i).value(j);
			} // Of for j
		} // Of for i

		// System.out.println("The input space is: " +
		// Arrays.deepToString(wholeX));
		// System.out.println("The output space is: " +
		// Arrays.toString(wholeY));
	}// Of the first constructor

	/**
	 ****************** 
	 * Setter.
	 * 
	 * @param paraInitial
	 *            The initial value.
	 * @param paraDecrement
	 *            The decrement value.
	 ****************** 
	 */
	public void setDistanceThresholds(double paraInitial, double paraDecrement) {
		distanceThresholdInitial = paraInitial;
		distanceThresholdDecrement = paraDecrement;
	}// Of setDistanceThresholds

	/**
	 ****************** 
	 * Randomize the training and testing sets.
	 * 
	 * @param paraTrainingFraction
	 *            The fraction of the training set.
	 ****************** 
	 */
	public void randomizeTrainingTesting(double paraTrainingFraction) {
		// Step 1. Randomize a sequence.
		int[] tempSequence = SimpleTools.getRandomOrder(wholeY.length);

		// Step 2. Copy the training set.
		int tempTrainingSize = (int) (wholeY.length * paraTrainingFraction);
		// For X, only space for references.
		trainingX = new double[tempTrainingSize][];
		trainingY = new double[tempTrainingSize][];
		for (int i = 0; i < tempTrainingSize; i++) {
			trainingX[i] = wholeX[tempSequence[i]];
			trainingY[i] = wholeY[tempSequence[i]];
		} // Of for i
		removalArray = new boolean[tempTrainingSize];

		// Step 3. Copy the testing set.
		int tempTestingSize = wholeY.length - tempTrainingSize;
		testingX = new double[tempTestingSize][];
		testingY = new double[tempTestingSize][];
		for (int i = 0; i < tempTestingSize; i++) {
			testingX[i] = wholeX[tempSequence[tempTrainingSize + i]];
			testingY[i] = wholeY[tempSequence[tempTrainingSize + i]];
		} // Of for i

		// System.out.println("The training X is: " +
		// Arrays.deepToString(trainingX));
		// System.out.println("The training Y is: " +
		// Arrays.deepToString(trainingY));
		// System.out.println("The testing X is: " +
		// Arrays.deepToString(testingX));
		// System.out.println("The testing Y is: " +
		// Arrays.deepToString(testingY));
	}// Of randomizeTrainingTesting

	/**
	 ****************** 
	 * Select data close to the hyperplane.
	 * 
	 * @param paraWeights
	 *            The hyperplane.
	 * @param paraDistance
	 *            The distance.
	 * @return The data indices.
	 ****************** 
	 */
	public int[] select(double[] paraWeights, double paraDistance) {
		int[] tempSelectionArray = new int[trainingX.length];
		double tempDistance = 0;
		int tempNumSelection = 0;
		for (int i = 0; i < trainingX.length; i++) {
			// This instance has been removed.
			if (removalArray[i]) {
				continue;
			} // Of if

			double tempPrediction = 0;
			for (int j = 0; j < trainingX[0].length; j++) {
				tempPrediction += paraWeights[j] * trainingX[i][j];
			} // Of for j
			tempDistance = Math.abs(tempPrediction - trainingY[i][0]);

			if (tempDistance > paraDistance) {
				tempSelectionArray[tempNumSelection] = i;
				tempNumSelection++;
			} // Of if
		} // Of for i

		System.out.println("" + tempNumSelection + " instances are far from the hyperplane.");

		// Compress
		int[] resultSelections = new int[tempNumSelection];
		for (int i = 0; i < resultSelections.length; i++) {
			resultSelections[i] = tempSelectionArray[i];
		} // Of for i

		return resultSelections;
	}// Of select

	/**
	 ****************** 
	 * Train the self-paced-regressor.
	 ****************** 
	 */
	public double[] train() {
		// Step 1. Build the original hyperplane.
		System.out.println("Training ... the training set has " + trainingX.length + " instances.");
		weights = train(trainingX, trainingY);

		// System.out.println("All data, the weights are: " +
		// Arrays.toString(weights));
		System.out.println(
				"The training error with all data is: " + computeError(trainingX, trainingY, weights));
		double tempError = computeError(testingX, testingY, weights);
		System.out.println("The testing error with all training data is: " + tempError);

		// Step 2. Increase distance gradually.
		double tempDistanceThreshold = distanceThresholdInitial;
		double tempNumNeighbors = 0;
		for (int i = 0; i < maxChangeLoops; i++) {
			// Step 2.1 Select a subset.
			int[] tempIndices = select(weights, tempDistanceThreshold);
			tempNumNeighbors = tempIndices.length;

			double[][] tempX = new double[tempIndices.length][];
			double[][] tempY = new double[tempIndices.length][];
			// System.out.println("TrainingY: " +
			// Arrays.deepToString(trainingY));
			// Copy data
			for (int j = 0; j < tempX.length; j++) {
				// System.out.print(", " + tempIndices[j] + ": " +
				// trainingY[tempIndices[j]][0]);
				tempX[j] = trainingX[tempIndices[j]];
				tempY[j] = trainingY[tempIndices[j]];
			} // Of for j

			// Step 2.2 Update the weights
			weights = train(tempX, tempY);

			// Step 2.3 Remove incorrectly classified instances.
			for (int j = 0; j < tempX.length; j++) {
				double tempPredict = innerProduct(tempX[j], weights);
				if (tempPredict > 0) {
					tempPredict = 1;
				} else {
					tempPredict = 0;
				} // Of if
					// tempValue = sigmoid(tempValue);
				if (tempPredict != tempY[j][0]) {
					System.out.println("Removing: " + j);
					removalArray[tempIndices[j]] = true;
				} // Of if
			} // Of for j

			// Step 2.3 Not all data are useful
			if (tempNumNeighbors > trainingX.length * instanceProportionThreshold) {
				// Enough training data are used.
				break;
			} // Of if

			// Step 2.4 Increase the distance threshold
			tempDistanceThreshold -= distanceThresholdDecrement;
			if (tempDistanceThreshold < 0.001) {
				break;
			} // Of if
		} // Of for i

		System.out.println("Finally, the threshold is " + tempDistanceThreshold + " with "
				+ tempNumNeighbors + " neighbors.");
		// System.out.println("The weights are: " + Arrays.toString(weights));
		System.out.println(
				"The training error with selected data is: " + computeError(trainingX, trainingY, weights));
		System.out.println(
				"The testing error with selected data is: " + computeError(testingX, testingY, weights));

		return weights;
	}// Of train

	/**
	 ****************** 
	 * Train with the given data matrices.
	 * 
	 * @param paraX
	 *            The input data.
	 * @param paraY
	 *            The output data.
	 * @return The weight vector.
	 ****************** 
	 */
	public double[] train(double[][] paraX, double[][] paraY) {
		// System.out.println("paraX: " + Arrays.deepToString(paraX));
		// System.out.println("paraY: " + Arrays.deepToString(paraY));
		double[][] tempWeights = new double[paraX[0].length][1];
		for (int i = 0; i < tempWeights.length; i++) {
			tempWeights[i][0] = 1;
		} // Of for i

		// Step 1. Construct a matrix object.
		Matrix tempX = new Matrix(paraX);
		Matrix tempY = new Matrix(paraY);
		Matrix tempW = new Matrix(tempWeights);
		Matrix tempSigma, tempError;

		// Step 2. Gradual ascendent.
		double alpha = 0.001;
		double maxCycles = 1000;
		for (int i = 0; i < maxCycles; i++) {
			tempSigma = sigmoid(tempX.times(tempW));
			tempError = tempY.minus(tempSigma);
			tempW = tempW.plus(tempX.transpose().times(tempError).times(alpha));
		} // Of for i

		double[] resultWeights = tempW.transpose().getArray()[0];

		// System.out.println("The new weights are: " +
		// Arrays.toString(resultWeights));

		return resultWeights;
	}// Of train

	/**
	 ****************** 
	 * Compute the sigmoid of the given value.
	 * 
	 * @param paraValue
	 *            The given value.
	 * @return The sigmoid value.
	 ****************** 
	 */
	public double sigmoid(double paraValue) {
		return 1.0 / (1 + Math.exp(-paraValue));
	}// Of sigmoid

	/**
	 ****************** 
	 * Compute the sigmoid of the given matrix.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 * @return The sigmoid values.
	 ****************** 
	 */
	public Matrix sigmoid(Matrix paraMatrix) {
		int tempRows = paraMatrix.getRowDimension();
		int tempColumns = paraMatrix.getColumnDimension();

		Matrix resultMatrix = new Matrix(tempRows, tempColumns);
		for (int i = 0; i < tempRows; i++) {
			for (int j = 0; j < tempColumns; j++) {
				double tempValue = paraMatrix.get(i, j);
				tempValue = sigmoid(tempValue);
				resultMatrix.set(i, j, tempValue);
			} // Of for j
		} // Of for i
		return resultMatrix;
	}// Of sigmoid

	/**
	 ****************** 
	 * Compute the error on the given set.
	 * 
	 * @param paraX
	 *            The input.
	 * @param paraY
	 *            The output.
	 * @param paraWeights
	 *            The weights.
	 * @return The error.
	 ****************** 
	 */
	public double computeError(double[][] paraX, double[][] paraY, double[] paraWeights) {
		double tempErrorSum = 0;

		for (int i = 0; i < paraX.length; i++) {
			double tempPredict = innerProduct(paraX[i], paraWeights);

			if (tempPredict > 0) {
				tempPredict = 1;
			} else {
				tempPredict = 0;
			} // Of if

			if (tempPredict != paraY[i][0]) {
				tempErrorSum++;
			} // Of if
		} // Of for i

		return tempErrorSum / paraX.length;
	}// Of computeError

	/**
	 ****************** 
	 * Compute the inner product of two arrays.
	 * 
	 * @param paraArray1
	 *            The first array.
	 * @param paraArray2
	 *            The second array.
	 * @return The error.
	 ****************** 
	 */
	public double innerProduct(double[] paraArray1, double[] paraArray2) {
		double resultValue = 0;
		for (int i = 0; i < paraArray1.length; i++) {
			resultValue += paraArray1[i] * paraArray2[i];
		} // Of for i

		return resultValue;
	}// Of innerProduct

	/**
	 ****************** 
	 * For integration test.
	 * 
	 * @param args
	 *            Not provided.
	 ****************** 
	 */
	public static void main(String args[]) {
		System.out.println("Starting self-paced regression ...");
		SelfPacedLogisticRegressor tempSelfPacedLogisticRegressor = new SelfPacedLogisticRegressor(
				"src/data/wdbc_norm_ex.arff");

		tempSelfPacedLogisticRegressor.randomizeTrainingTesting(0.6);
		double[] tempWeights = tempSelfPacedLogisticRegressor.train();
	}// Of main
}// Of class SelfPacedLogisticRegressor
