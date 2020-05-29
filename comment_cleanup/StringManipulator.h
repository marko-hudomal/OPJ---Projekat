#pragma once
#ifndef _STRING_MANIPULATOR_H_

#define _STRING_MANIPULATOR_H_

class StringManipulator {

public:

	std::string cleanUpText(std::string&);
	std::string trimBlanks(const std::string&);

	const std::string& getWhitespace() { return kWhitespace; }
	const std::string& getIgnoredSymbols() { return kIgnoredSymbols; }

private:

	std::string kWhitespace = " \n\r\t\f\v\\n";
	std::string kIgnoredSymbols = "*|-";

	std::string trimStart(const std::string&);
	std::string trimEnd(const std::string&);
	std::string correctDoubleSpaces(std::string&);

	bool twoSpacesAtPosition(std::string&, int);
};

#endif