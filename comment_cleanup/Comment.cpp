#include <fstream>
#include <algorithm>
#include <string>

#include "Comment.h"

void Comment::cleanUpCommentText() {
	std::string text = content[kCommentText];
	std::string modifiedText;

	// Circumvent the filter that deletes trailing 'n' characters
	// as part of ignoring '\n' trailing newlines.
	if (text[text.length() - 1] == 'n') {
		text += '.';
	}

	text = stringManipulator.trimBlanks(text);

	for (size_t i = 0; i < text.length(); i++) {
		if (isIgnoredSymbolAtPosition(text, i)) {

			if (newLineAtPosition(text, i + 1)) {
				// Ignore lines that only contain ignored symbols and whitespace.
				modifiedText += "\\n";
				i += 3;
			}
			else {
				if (isSingleDashAtPosition(text, i)) {
					modifiedText += text[i];
				}
				else {
					i = ignoreInvalidSymbolsAndWhitespaceFromPosition(text, i + 1);
				}
			}
		}
		else {
	
			// Check for a new-line character denoted in the .tsv as three spaces.
			if (newLineAtPosition(text, i)) {
				modifiedText += "\\n";
				i += 2;
			}
			else {
				modifiedText += text[i];
			}
		}
	}

	// Clean up resulting text.
	modifiedText = stringManipulator.cleanUpText(modifiedText);

	content[kCommentText] = modifiedText;
}

bool Comment::newLineAtPosition(std::string& text, int i) {
	return text[i] == ' ' && text[i + 1] == ' ' && text[i + 2] == ' ';
}

bool Comment::isIgnoredSymbolAtPosition(std::string& text, int i) {
	// Returns true if an ignored symbol is located at text[i].
	std::string ignoredSymbols = stringManipulator.getIgnoredSymbols();
	return ignoredSymbols.find(text[i]) != std::string::npos; 
}

bool Comment::isSingleDashAtPosition(std::string& text, int i) {
	return text[i] == '-' && text[i + 1] != '-';
}
size_t Comment::ignoreInvalidSymbolsAndWhitespaceFromPosition(std::string& text, int i) {
	std::string ignoredCharacters = stringManipulator.getWhitespace() + stringManipulator.getIgnoredSymbols();
	return text.find_first_not_of(ignoredCharacters, i) - 1;
}

