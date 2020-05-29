#include <string>
#include <algorithm>

#include "StringManipulator.h"

std::string StringManipulator::cleanUpText(std::string& text) {

	std::string cleanText = correctDoubleSpaces(text);
	cleanText = trimBlanks(cleanText);

	return cleanText;
}

std::string StringManipulator::correctDoubleSpaces(std::string& text) {
	std::string correctedText;
	for (size_t i = 0; i < text.length() - 1; i++) {
		if (twoSpacesAtPosition(text, i)) {
			correctedText += "\\n";
			i++;
		}
		else {
			correctedText += text[i];
		}
	}

	if (text[text.length() - 1] != ' ') {
		correctedText += text[text.length() - 1];
	}

	return correctedText;
}

std::string StringManipulator::trimBlanks(const std::string& text) {
	return trimEnd(trimStart(text));
}

std::string StringManipulator::trimEnd(const std::string& text) {
	size_t end = text.find_last_not_of(kWhitespace);
	return (end == std::string::npos) ? "" : text.substr(0, end + 1);
}

std::string StringManipulator::trimStart(const std::string& text) {
	size_t start = text.find_first_not_of(kWhitespace);
	return (start == std::string::npos) ? "" : text.substr(start);
}

bool StringManipulator::twoSpacesAtPosition(std::string& text, int i) {
	return text[i] == ' ' && text[i + 1] == ' ';
}

