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

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws IOException {
    	/*
		 * VFSListDataset<FImage> scenes = new
		 * VFSListDataset<FImage>("/Users/tangke/Desktop/data/training",
		 * ImageUtilities.FIMAGE_READER); System.out.print(scenes.size());
		 * 
		 * DisplayUtilities.display("ATT Scenes", scenes);
		 */

		VFSGroupDataset<FImage> trainScenes = new VFSGroupDataset<FImage>("/Users/maggie/Desktop/data/training",
				ImageUtilities.FIMAGE_READER);
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainScenes, 60, 0, 40);

//		for (final Entry<String, VFSListDataset<FImage>> entry : groupedScenes.entrySet()) {
//			DisplayUtilities.display(entry.getKey(), entry.getValue());
//		}
		
		TinyImage featureExtractor = new TinyImage();
		
		int k = 18;
		
		KNNAnnotator<FImage, String, FloatFV> knn = new KNNAnnotator<>(featureExtractor, FloatFVComparison.EUCLIDEAN, k);
		
		knn.train(splits.getTrainingDataset());
		System.out.println("Training finished");
		
		VFSListDataset<FImage> testScenes = new VFSListDataset<>("/Users/maggie/Desktop/data/testing", ImageUtilities.FIMAGE_READER);
		
		FileObject[] files = testScenes.getFileObjects();
		
		PrintWriter pw = new PrintWriter("run1.txt");
		String str;
		for(int i=0; i<testScenes.size(); i++) {
			str = files[i].getName().getBaseName() + " " + knn.classify(testScenes.get(i)).getPredictedClasses().iterator().next();
			pw.println(str);
		}
		
		pw.close();
		
		  /**
//       * Run test dataset
//       */
      float correct = 0;
      float incorrect = 0;
      for (Map.Entry<String, ListDataset<FImage>> entry : splits.getTestDataset().entrySet()) {
          for (FImage test : entry.getValue()) {
              String predicted = knn.classify(test).getPredictedClasses().iterator().next();
              String actual = entry.getKey();
              if (predicted.equals(actual)) {
                  correct += 1;
              } else incorrect += 1;
          }
      }
      float accuracy = 0f;
      System.out.println();
      accuracy += correct / (correct + incorrect) * 100;
//  }
  System.out.println("Accuracy:" + accuracy);
		
//		featureExtractor.extractFeature(ImageUtilities.readF(new File("/Users/maggie/Desktop/data/training/bedroom/0.jpg")));
		
    }
    
    
}
