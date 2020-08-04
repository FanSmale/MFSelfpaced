package algorithm;

import java.util.Arrays;

/**
 * A toy example. <br>
 * Project: Self-paced learning.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/MFAdaBoosting.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 *         Data Created: July 26, 2020.<br>
 *         Last modified: July 26, 2020.
 * @version 1.0
 */

public class ToyExample {

	/**
	 ****************** 
	 * For integration test.
	 * 
	 * @param args
	 *            Not provided.
	 ****************** 
	 */
	public static int[] spld(double[] paraLossArray, int[] paraGroupMembership,
			double paraLambda, double paraGamma) {
		// Step 1. Count the number of groups. We assume that the group number
		// is from 1 to k.
		int tempNumGroups = -1;
		for (int i = 0; i < paraGroupMembership.length; i++) {
			if (tempNumGroups < paraGroupMembership[i]) {
				tempNumGroups = paraGroupMembership[i];
			}// Of if
		}// Of for i
		System.out.println("tempNumGroups = " + tempNumGroups);

		int[] tempGroupIndices = new int[tempNumGroups];
		for (int i = 0; i < tempGroupIndices.length; i++) {
			tempGroupIndices[i] = i + 1;
		}// Of for i
		System.out.println("tempGroupIndices = " + Arrays.toString(tempGroupIndices));
		// R: groupidx = unique(groupmembership)

		// Step 2. Initialize selected indices and respective score.
		boolean[] tempSelectedIndices = new boolean[paraLossArray.length];
		double[] tempSelectedScores = new double[paraLossArray.length];

		// Step 3. Select in each group.
		for (int i = 0; i < tempNumGroups; i++) {
			// Step 3.1 Form the current group.
			int[] tempInGroupIndices = which(paraGroupMembership, tempGroupIndices[i]);
			System.out.println("tempInGroupIndices = " + Arrays.toString(tempInGroupIndices));
			int tempGroupSize = tempInGroupIndices.length;

			// Step 3.2 Copy respective loss.
			double[] tempInGroupLossArray = new double[tempGroupSize];
			for (int j = 0; j < tempInGroupLossArray.length; j++) {
				tempInGroupLossArray[j] = paraLossArray[tempInGroupIndices[j]];
			}// Of for j

			// Step 3.3 Rank the loss in the current group in ascendant order. 
			System.out.println("tempInGroupLossArray = "
					+ Arrays.toString(tempInGroupLossArray));
			int[] tempInGroupRankArray = rankAscendant(tempInGroupLossArray);
			System.out.println("tempInGroupRankArray = "
					+ Arrays.toString(tempInGroupRankArray));

			//Step 3.4 Determine which ones in the current group to choose.
			for (int j = 0; j < tempGroupSize; j++) {
				if (tempInGroupLossArray[j] < paraLambda
						+ paraGamma
						/ (Math.sqrt(tempInGroupRankArray[j]) + Math
								.sqrt(tempInGroupRankArray[j] - 1))) {
					tempSelectedIndices[tempInGroupIndices[j]] = true;
					System.out.println("Selecting: " + tempInGroupIndices[j]);
				}// Of if
				tempSelectedScores[tempInGroupIndices[j]] = tempInGroupLossArray[j]
						- paraLambda
						- paraGamma
						/ (Math.sqrt(tempInGroupRankArray[j]) + Math
								.sqrt(tempInGroupRankArray[j] - 1));
			}// Of for j
		}// Of for i

		System.out.println("New scores: " + Arrays.toString(tempSelectedScores));
		int[] resultSelectedIndices = which(tempSelectedIndices);
		System.out.println("Indices: " + Arrays.toString(resultSelectedIndices));

		return resultSelectedIndices;
	}// Of spld

	/**
	 ****************** 
	 * Compress an array.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] compressArray(int[] paraArray, int paraLength) {
		int[] resultArray = new int[paraLength];
		for (int i = 0; i < resultArray.length; i++) {
			resultArray[i] = paraArray[i];
		}// Of for i
		return resultArray;
	}// Of compressArray

	/**
	 ****************** 
	 * Rank the array. The biggest value will be 1.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] rankAscendant(double[] paraArray) {
		int[] resultArray = new int[paraArray.length];
		for (int i = 0; i < paraArray.length; i++) {
			resultArray[i] = 1;
			for (int j = 0; j < paraArray.length; j++) {
				if (paraArray[j] < paraArray[i]) {
					resultArray[i]++;
				}// Of if

				if (paraArray[j] == paraArray[i]) {
					if (j < i) {
						resultArray[i]++;
					}// Of if
				}// Of if
			}// Of for j
		}// Of for i
		return resultArray;
	}// Of compressArray

	/**
	 ****************** 
	 * Which elements in the array are equal to the given value.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] which(int[] paraArray, int paraValue) {
		int[] tempArray = new int[paraArray.length];

		int tempCounter = 0;
		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] == paraValue) {
				tempArray[tempCounter] = i;
				tempCounter++;
			}// Of if
		}// Of for i

		int[] resultArray = compressArray(tempArray, tempCounter);

		return resultArray;
	}// Of which

	/**
	 ****************** 
	 * Which elements in the array are true.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] which(boolean[] paraArray) {
		int[] tempArray = new int[paraArray.length];

		int tempCounter = 0;
		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i]) {
				tempArray[tempCounter] = i;
				tempCounter++;
			}// Of if
		}// Of for i

		int[] resultArray = compressArray(tempArray, tempCounter);

		return resultArray;
	}// Of which

	/**
	 ****************** 
	 * Obtain the subset of a char set
	 * 
	 * @param paraArray
	 *            The given char array.
	 ****************** 
	 */
	public static char[] charSubset(char[] paraArray, int[] paraIndices) {
		char[] resultArray = new char[paraIndices.length];
		for (int i = 0; i < paraIndices.length; i++) {
			resultArray[i] = paraArray[paraIndices[i]];
		}// Of for i

		return resultArray;
	}// Of charSubset

	/**
	 ****************** 
	 * For integration test.
	 * 
	 * @param args
	 *            Not provided.
	 ****************** 
	 */
	public static void main(String args[]) {
		char[] tempCharSet = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
				'l', 'm', 'n' };
		int[] tempGroupMembership = { 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4 };
		double[] tempLossArray = { 0.05, 0.12, 0.12, 0.12, 0.15, 0.40, 0.17, 0.18, 0.35,
				0.15, 0.16, 0.20, 0.50, 0.28 };

		// System.out.println("When paraLambda=0.15, SPL selects:");
		// System.out.println(spl(paraLossArray, 0.15));
		int[] tempSelections = null;
		tempSelections = spld(tempLossArray, tempGroupMembership, 0.05, 0.2);
		char[] tempSelectionsInChars = charSubset(tempCharSet, tempSelections);

		String tempResult = Arrays.toString(tempSelectionsInChars);
		System.out.println("*****************");
		System.out.println("When lambda = 0.05 and gamma = 0.2, SPLD selects: "
				+ tempResult);

		tempSelections = spld(tempLossArray, tempGroupMembership, 0.00, 0.285);
		tempSelectionsInChars = charSubset(tempCharSet, tempSelections);
		tempResult = Arrays.toString(tempSelectionsInChars);
		System.out.println("*****************");
		System.out.println("When lambda = 0.0 and gamma = 0.285, SPLD selects: "
				+ tempResult);
	}// Of main

}// Of class ToyExample

