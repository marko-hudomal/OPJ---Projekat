package extractor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class Extractor {
	private static String suffix = ".php";
	private static String programmingLanguage = "php";
	private static String naturalLanguage = "EN";

	private static String outFilePathCsv = "comments.csv";
	private static String outFilePathResourcesCsv = "resources.csv";
	private static File outFileCsv;
	private static File outFileResourcesCsv;
	private static PrintWriter writerCsv;
	private static PrintWriter writerResourcesCsv;
	private static String separatorCsv = ",";
	private static String rootDirName;

	private static boolean inComment;

	// 0 - no
	// 1 - now (this line)
	// 2 - before (previous line)
	private static int prevInlineComment;

	private static StringBuilder makeCommentColumns(String filePath, int line, String separator) {
		filePath = filePath.split(rootDirName)[1];
		StringBuilder sb = new StringBuilder();
		sb.append(naturalLanguage);
		sb.append(separator);
		sb.append(programmingLanguage);
		sb.append(separator);
		// System.out.println(filePath);
		String dirs[] = filePath.split("[\\\\]");
		sb.append(dirs[2].split("-")[0]);
		sb.append(separator);

		for (int i = 3; i < dirs.length; i++) {
			sb.append(dirs[i]);
			if (i < dirs.length - 1) {
				sb.append("/");
			}
		}
		sb.append(separator);

		sb.append(dirs[2].split("-")[0]);
		sb.append("/");
		for (int i = 3; i < dirs.length; i++) {
			sb.append(dirs[i]);
			sb.append("/");
		}
		sb.append(line);
		sb.append(separator);

		if (separator == separatorCsv) {
			sb.append("\"");
		}
		return sb;
	}

	private static void writeComment(StringBuilder singleCommentCsv) {
		singleCommentCsv.append("\"");
		singleCommentCsv.append("\r\n");
		writerCsv.write(singleCommentCsv.toString());
	}

	private static void writeResourceForFile(File file, int lines) {

		StringBuilder sb = new StringBuilder();
		String filePath = file.getPath().split(rootDirName)[1];

		String dirs[] = filePath.split("[\\\\]");
		sb.append(dirs[2].split("-")[0]);
		sb.append(separatorCsv);

		for (int i = 3; i < dirs.length; i++) {
			sb.append(dirs[i]);
			if (i < dirs.length - 1) {
				sb.append("/");
			}
		}
		sb.append(separatorCsv);

		sb.append(lines);
		sb.append(separatorCsv);

		// Make url
		sb.append("github.com/");
		sb.append(dirs[1]);
		sb.append("/");
		sb.append(dirs[2].split("-")[0]);
		sb.append("/tree/");
		sb.append(dirs[2].split("-")[1]);
		for (int i = 3; i < dirs.length; i++) {
			sb.append("/");
			sb.append(dirs[i]);
		}

		sb.append("\r\n");
		writerResourcesCsv.write(sb.toString());
	}

	private static void readFile(File file) {
		inComment = false;
		prevInlineComment = 0;
		int lineNum = 1;
		boolean inStringInCode;
		StringBuilder singleCommentCsv = null;
		boolean commentFound = false;
		Character apostrophe = ' ';
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			for (String line; (line = br.readLine()) != null;) {
				// Skip html file
				if (line.toLowerCase().contains("<html")) {
					break;
				}
				inStringInCode = false;
				for (int i = 0; i < line.length(); i++) {
					if (!inComment && (line.charAt(i) == '\'' || line.charAt(i) == '"')) {
						if (!inStringInCode) {
							apostrophe = line.charAt(i);
							inStringInCode = true;
						} else if (line.charAt(i) == apostrophe) {
							inStringInCode = false;
						}

					}
					if (inStringInCode) {
						continue;
					}
					if (i < line.length() - 1 && !inComment && line.charAt(i) == '/' && line.charAt(i + 1) == '*') {
						if (prevInlineComment == 2) {
							prevInlineComment = 0;
							writeComment(singleCommentCsv);
						}
						inComment = true;
						commentFound = true;
						singleCommentCsv = makeCommentColumns(file.getPath(), lineNum, separatorCsv);

						i++;
						continue;
					} else if (i < line.length() - 1 && !inComment && line.charAt(i) == '/'
							&& line.charAt(i + 1) == '/') {
						if (prevInlineComment == 0) {
							singleCommentCsv = makeCommentColumns(file.getPath(), lineNum, separatorCsv);
						}
						inComment = true;
						commentFound = true;
						prevInlineComment = 1;
						i++;
						continue;
					} else if (!inComment && line.charAt(i) == '#') {
						if (prevInlineComment == 0) {
							singleCommentCsv = makeCommentColumns(file.getPath(), lineNum, separatorCsv);
						}
						inComment = true;
						commentFound = true;
						prevInlineComment = 1;
						i++;
						continue;
					} else if (prevInlineComment == 0 && i < line.length() - 1 && line.charAt(i) == '*'
							&& line.charAt(i + 1) == '/') {
						writeComment(singleCommentCsv);
						inComment = false;
						i++;
						continue;
					}
					if (line.charAt(i) != ' ' && line.charAt(i) != '\t') {
						if (prevInlineComment == 2) {
							prevInlineComment = 0;
							writeComment(singleCommentCsv);
						}
					}
					if (inComment) {
						if (line.charAt(i) == '"') {
							singleCommentCsv.append('\"');
							singleCommentCsv.append('\"');
						} else {
							singleCommentCsv.append(line.charAt(i));
						}
					}
				}
				if (inComment) {
					singleCommentCsv.append("\n");
				}
				if (prevInlineComment == 1) {
					prevInlineComment = 2;
					inComment = false;
				}
				lineNum++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		if (commentFound) {
			writeResourceForFile(file, lineNum - 1);
		}
	}

	private static void listDirAndFindFiles(String dirPath) {
		File rootDir = new File(dirPath);

		for (File file : rootDir.listFiles()) {
			if (file.isDirectory()) {
				listDirAndFindFiles(file.getAbsolutePath());
			} else if (file.getName().endsWith(suffix)) {
				readFile(file);
			}
		}
	}

	public static void main(String args[]) {
		long t = System.currentTimeMillis();
		/* Dir hierarchy:
		 * 
 		 * 	rootDir
		 * 		gitUserDir
		 * 			repoDir
		 * 			repoDir
		 * 			...
		 * 		gitUserDir
		 * 			repoDir
		 * 			...
		 * 		...
		 *
		 * repoDir name = repoName-repoBranch
		 * When you zip download repository and then extract you will get it like this.
		*/
		String rootDirPath = "D:\\faks\\master\\opj\\repositories";

		outFileCsv = new File(rootDirPath + File.separator + outFilePathCsv);
		if (!outFileCsv.exists()) {
			try {
				outFileCsv.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}

		outFileResourcesCsv = new File(rootDirPath + File.separator + outFilePathResourcesCsv);
		if (!outFileResourcesCsv.exists()) {
			try {
				outFileResourcesCsv.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}

		try {
			writerCsv = new PrintWriter(outFileCsv);
			writerResourcesCsv = new PrintWriter(outFileResourcesCsv);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
		File f = new File(rootDirPath);
		rootDirName = f.getName();
		listDirAndFindFiles(f.getAbsolutePath());

		writerCsv.flush();
		writerCsv.close();
		writerResourcesCsv.flush();
		writerResourcesCsv.close();

		t = System.currentTimeMillis() - t;
		t /= 1000;
		System.out.println("Extracting finished successfully in " + t + " sec!");
	}
}
