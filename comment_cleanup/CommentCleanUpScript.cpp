#include <iostream>
#include <fstream>
#include <string>
#include "Comment.h"

void cleanUpAndSaveToOutput(std::string inputFileName, std::string outputFileName) {
	
	std::ifstream inputFile;	
	inputFile.open(inputFileName);

	if (!inputFile.is_open()) {
		std::cout << "Error while opening input file." << std::endl;
		return;
	}

	std::ofstream outputFile;
	outputFile.open(outputFileName);

	if (!outputFile.is_open()) {
		std::cout << "Error while opening output file." << std::endl;
		return;
	}
	
	Comment currentComment;
	std::string line;

	// Copy the first line.
	getline(inputFile, line);
	outputFile << line << std::endl;
	
	while (!inputFile.eof()) {

		currentComment = Comment();

		inputFile >> currentComment;
		currentComment.cleanUpCommentText();
		outputFile << currentComment;
	}

	inputFile.close();
	outputFile.close();
}

int main() {
	
	const std::string kInputFileName = "input.tsv";
	const std::string kOutputFileName = "output.tsv";

	cleanUpAndSaveToOutput(kInputFileName, kOutputFileName);
	
	return 0;
}