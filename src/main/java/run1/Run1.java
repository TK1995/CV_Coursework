package run1;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Map;

import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

public class Run1 {
	public static void main(String[] args) throws IOException {

		VFSGroupDataset<FImage> trainScenes = new VFSGroupDataset<FImage>("/Users/maggie/Desktop/data/training",
				ImageUtilities.FIMAGE_READER);
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainScenes, 75, 0, 25);

		TinyImage featureExtractor = new TinyImage();

		int k = 16;

		KNNAnnotator<FImage, String, FloatFV> knn = new KNNAnnotator<>(featureExtractor, FloatFVComparison.EUCLIDEAN,
				k);

		knn.train(splits.getTrainingDataset());
		System.out.println("Training finished");

		VFSListDataset<FImage> testScenes = new VFSListDataset<>("/Users/maggie/Desktop/data/testing",
				ImageUtilities.FIMAGE_READER);

		FileObject[] files = testScenes.getFileObjects();

		PrintWriter pw = new PrintWriter("run1.txt");
		String str;
		for (int i = 0; i < testScenes.size(); i++) {
			str = files[i].getName().getBaseName() + " "
					+ knn.classify(testScenes.get(i)).getPredictedClasses().iterator().next();
			pw.println(str);
		}

		// close the stream
		pw.close();

		// test and get the accuracy
		float correct = 0;
		float incorrect = 0;
		for (Map.Entry<String, ListDataset<FImage>> entry : splits.getTestDataset().entrySet()) {
			for (FImage testImg : entry.getValue()) {
				String result = knn.classify(testImg).getPredictedClasses().iterator().next();
				if (result.equals(entry.getKey())) {
					correct++;
				} else
					incorrect++;
			}
		}
		float accuracy = correct / (correct + incorrect) * 100;

		System.out.println("Accuracy: " + accuracy);
	}

}
