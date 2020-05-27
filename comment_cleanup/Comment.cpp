#include <fstream>

#include "Comment.h"

void Comment::cleanUpCommentText() {
	std::string text = content[kCommentText];

	std::string modifiedText;

	for (size_t i = 0; i < text.length(); i++) {
		if (text[i] != '*') {

			// Check for a new-line character denoted in the .tsv as three spaces.
			if (newLineAtPosition(text, i)) {
				modifiedText += "\\n";
				i += 2;
			}
			else {
				modifiedText += text[i];
			}
		}
		else {
			// Check if the line only contains an asterisk.
			if (newLineAtPosition(text, i + 1)) {
				modifiedText += "\\n";
				i += 3;
			}
			else {
				// Ignore the extra space after every asterisk.
				i++;
			}
			
		}
	}

	content[kCommentText] = modifiedText;
}

bool Comment::newLineAtPosition(std::string& text, int i) {
	return text[i] == ' ' && text[i + 1] == ' ' && text[i + 2] == ' ';
}