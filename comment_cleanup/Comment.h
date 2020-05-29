#pragma once
#ifndef _COMMENT_H_
#define _COMMENT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "StringManipulator.h"

class Comment {

public:

	void cleanUpCommentText();

	friend std::ifstream& operator>>(std::ifstream& stream, Comment& comment) {

		std::string line;
		getline(stream, line);

		size_t position, previousPosition;
		
		position = 0;
		while (position <= line.length()) {
			previousPosition = position;
			position = line.find('\t', previousPosition);

			if (position == std::string::npos) {
				position = line.length();
			}

			std::string value = line.substr(previousPosition, position - previousPosition);
			comment.content.push_back(value);

			position++;
		}
		return stream;
	}

	friend std::ofstream& operator<<(std::ofstream& stream, const Comment& comment) {

		std::string result;

		for (int i = 0; i <= (int)kCommentClass; i++) {
			result += comment.content[i];
			
			if (i != (int)kCommentClass) {
				result += '\t';
			}
		}

		stream << result << std::endl;
		return stream;
	}

private:

	enum Section {
		kNaturalLanguageID,
		kProgrammingLanguage,
		kRepoID,
		kSourceID,
		kCommentID,
		kCommentText,
		kCommentClass
	};

	std::vector<std::string> content;
	StringManipulator stringManipulator;

	bool newLineAtPosition(std::string&, int);
	bool isIgnoredSymbolAtPosition(std::string&, int);
	bool isSingleDashAtPosition(std::string&, int);
	size_t ignoreInvalidSymbolsAndWhitespaceFromPosition(std::string&, int);
};

#endif