package FileSorter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Comparator;
import java.io.FileNotFoundException;
import java.io.IOException;

public class FileSorter {
	
	public void sortFile(String inputFile) throws FileNotFoundException, IOException {
		FileReader fileReader = new FileReader(inputFile);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		String inputLine;
		List<String> lineList = new ArrayList<String>();
		while ((inputLine = bufferedReader.readLine()) != null) {
			lineList.add(inputLine);
		}
		fileReader.close();
		
		Collections.sort(lineList, new Comparator<String>() {
		    public int compare(String str1, String str2) {
		        return extractInt(str1) - extractInt(str2);
		    }
		    int extractInt(String s) {
		        String num = s.replaceAll("\\D", "");
		        return num.isEmpty() ? 0 : Integer.parseInt(num); // if no digits found, return 0
		    }
		});

		FileWriter fileWriter = new FileWriter(inputFile);
		PrintWriter out = new PrintWriter(fileWriter);
		for (String outputLine : lineList) {
			out.println(outputLine);
		}
		out.flush();
		out.close();
		fileWriter.close();
	}
}