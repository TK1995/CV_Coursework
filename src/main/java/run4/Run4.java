package run4;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileObject;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.identity.Identifiable;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.AbstractAnnotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import FileSorter.FileSorter;

public class Run4 {
	
	private VFSGroupDataset<FImage> trainingData; //The training data
	private VFSGroupDataset<FImage> testingData; //The testing data
	
	String filename = "run3.txt";
	
	public static void main(String args[]) {
		Run4 run = new Run4();
		run.performClassification();
	}
	
	//The run three constructor, initialises/loads both the training and testing data.
	Run4() {
		try {
			trainingData = new VFSGroupDataset<FImage>("/Users/tangke/Desktop/data/training", ImageUtilities.FIMAGE_READER);
			testingData = new VFSGroupDataset<FImage>("/Users/tangke/Desktop/data/testing", ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
	}
	
	//The main method which performs the classification process
	public void performClassification() {
		//Converts the VFSGroupDataset for training into a MapBackedDataset, by turning each FImage into a wrapper (Record), containing the image and the name
		MapBackedDataset<String, ListDataset<Record>, Record> training = transformDataset(getTrainingData());
		
		//Creates a dense sift with a step size of 5, and bin size of 7.
		DenseSIFT dsift = new DenseSIFT(5, 7);
		
		System.out.println("Extracting Features...");
		
		//Uses the trainQuantiser method to obtain a HardAssigner, named assigner, using the training a sample of the training data and the dense sift
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 30), dsift);
		
		//Creates a HomogeneousKernelMap instance
		HomogeneousKernelMap kernelmap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		//Creates the new Extractor class object with the given dense sift and assigner, warpping it in the HomogeneousKernelMap to give a FeatureExtractor
		FeatureExtractor<DoubleFV, Record> extractor = kernelmap.createWrappedExtractor(new Extractor(dsift, assigner));
		
		System.out.println("Training the classifier...");
		
		//Uses a NaiveBayesAnnotator (classifier) with the given extractor and mode of operation
		NaiveBayesAnnotator<Record, String> ann = new NaiveBayesAnnotator<Record, String>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
		//Start training the classifier
		ann.train(training);
		
		System.out.println("Testing Accuracy...");
		
		//Used to test the accuracy of the classifier on the seen data
		testAccuracy(training, ann);
		
		System.out.println("Outputting results to file...");
		
		//Create a map to store the results to
		HashMap<Integer, String> results = new HashMap<Integer, String>();
		
		//Go through each category in the testing data - There is no categories here, so it'll only run once.
		for(String category : getTestingData().getGroups()) {
			try {
				//Get all the images in the category - here it'll just get all the images in testing
				ListDataset<FImage> testingImages = getTestingData().get(category);
				//Get all the fileNames in the testing dataset
				List<String> fileNames = getFileNames(getTestingData().getFileObject(category).getChildren());
				
				//If there are valid files and its the expected size, we can continue to classify
				if(!fileNames.isEmpty() && fileNames.size() == testingImages.size()) {
					for(int i = 0; i<testingImages.size(); i++) {
						//Classify the record/image, and return the results
						ClassificationResult<String> result = ann.classify(new Record(fileNames.get(i), testingImages.get(i)));
						String resultString = null;
						double maxConfidence = -1.0;
						//Go through each result's predicted class, and find the one with the highest confidence
						for(String s : result.getPredictedClasses()) {
							double confidence = result.getConfidence(s);
							if(confidence > maxConfidence) {
								resultString = s;
								maxConfidence = confidence;
							}
						}
						//System.out.println(total + " items processed");
						//Add to results, the image (its filename) and the class with the classifier predicts it most likely belongs to
						results.put(Integer.parseInt(fileNames.get(i).split("\\.")[0]), resultString);
					}
				}
			} catch (FileSystemException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		//Finally, save the results (in order) to the run3.txt file.
		try {
			File file = new File(filename);
			FileWriter fileWriter = new FileWriter(file);
		    PrintWriter writer = new PrintWriter(fileWriter);
			for(Entry<Integer, String> entry : results.entrySet()) {
				writer.println(String.valueOf(entry.getKey()) + ".jpg " + entry.getValue());
			}
			writer.close();
			fileWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Finished outputting to file!");
		
		// sort output file lines in numerical order
		FileSorter fs = new FileSorter();
		fs.sortFile(filename);
	}
	
	//Method for testing the classifier's performance, will take a sample of training images and attempt to classify them.
	//As the sample of training images categories are known, we can compare to see the correctness of the classifier for seen data.
	//@param: MapBackedDataset<String, ListDataset<Record>, Record> training - The training data
	//@param: AbstractAnnotator<Record, String> annotator - The annotator / classifier
	private void testAccuracy(MapBackedDataset<String, ListDataset<Record>, Record> training, AbstractAnnotator<Record, String> annotator) {
		float correct = 0;
		float total = 0;
		//Splits the training data into training and testing data.
		GroupedRandomSplitter<String, Record> splits = new GroupedRandomSplitter<String, Record>(training, 15, 0, 15);
		//Gets the testing data from the split training data.
		GroupedDataset<String, ListDataset<Record>, Record> test = splits.getTestDataset();
		//For loop iterating through each group in the testing data
		for(String group : test.getGroups()) {
			//Gets all records for the group which is currently being looked at
			ListDataset<Record> testingImages = test.get(group);
			
			//For each record in that record, classify it and get pick the result with highest confidence.
			for(Record r : testingImages) {
				ClassificationResult<String> result = annotator.classify(r);
				String resultString = null;
				double maxConfidence = -1.0;
				//Loop through all the results, and pick the one with the highest confidence.
				for(String s : result.getPredictedClasses()) {
					double confidence = result.getConfidence(s);
					if(confidence > maxConfidence) {
						resultString = s;
						maxConfidence = confidence;
					}
				}
				//Print out the result for that record, along with the prediction of the class with highest confidence.
				System.out.println(r.getID() + ".jpg" + ", Actual: " + group + ", Prediction: " + resultString);
				//If it's correct, add it to the correct total, and regardless add to the total total.
				if(group.equalsIgnoreCase(resultString))
					correct++;
				total++;
			}
		}
		//Used to output the accuracy to 2 decimal places
		DecimalFormat df2 = new DecimalFormat("#.##");
		//Outputs the accuracy of the classifier for the seen data.
		System.out.println("Accuracy: " + df2.format((correct/total)*100) + "%");
	}
	
	//Transforms a VFSGroupDataset dataset into a MapBackedDataset, converting the images into records.
	public static MapBackedDataset<String, ListDataset<Record>, Record> transformDataset(VFSGroupDataset<FImage> data) {
		//Construct a new MapBackedDataset
		MapBackedDataset<String, ListDataset<Record>, Record> dataset = new MapBackedDataset<String, ListDataset<Record>, Record>();
		
		for(String category : data.getGroups()) {
			//List of images of the current category
			ListDataset<FImage> categoryImages = data.get(category);
			//Put that category and its images into the mapped data set
			ListDataset<Record> recordList = new ListBackedDataset<Record>();
			
			try {
				//Call the getFileNames method to get all the file names in the category we're currently looking at
				List<String> fileNames = getFileNames(data.getFileObject(category).getChildren());
				
				//For each image in the category, make a corresponding Record of it's image and ID (from fileNames) and add it to the recordList ListDataset
				if(!fileNames.isEmpty() && fileNames.size() == categoryImages.size()) {
					for(int i = 0; i<categoryImages.size(); i++) {
						recordList.add(new Record(fileNames.get(i), categoryImages.get(i)));
					}
				}
				
				//If the dataset isn't empty, add it to the ListDataset dataset
				if(!recordList.isEmpty())
					dataset.put(category, recordList);
			} catch(IOException e) {
				e.printStackTrace();
			}
		}
		//Return the dataset
		return dataset;
	}
	
	//Simple method for returning all the file names in the given FileObject[] fileObj, provided it is a .jpg file.
	private static List<String> getFileNames(FileObject[] fileObj) {
		//Construct a new ArrayList
		List<String> fileNames = new ArrayList<String>();
		//For all items in the fileObject
		for(int i = 0; i<fileObj.length; i++) {
			//If it's a jpg image, add it to the arraylist
			if(fileObj[i].getName().getBaseName().contains(".jpg"))
				fileNames.add(fileObj[i].getName().getBaseName());
		}
		//Return the fileNames arraylist with all the file names in it
		return fileNames;
	}
	
	//Getter method for getting the training data
	public VFSGroupDataset<FImage> getTrainingData() {
		return trainingData;
	}
	
	//Getter method for getting the testing data
	public VFSGroupDataset<FImage> getTestingData() {
		return testingData;
	}
	
	//Performs K-means clustering on the sample of SIFT features, which build the HardAssigner which assigns features to identifiers
	//This method has been inspired from the OpenImaj tutorial 12, for classification.
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<Record> sample, DenseSIFT dsift) {
		//Create a new list of a LocalFeatureList of ByteDSIFTKeypoint
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		//For each record in the sample
		for (Record rec : sample) {
			//Get the image
		    FImage img = rec.getImage();
		    //Analyse the image
		    dsift.analyseImage(img);
		    //Add it to the allKeys list
		    allkeys.add(dsift.getByteKeypoints(0.005f));
		}
		
		//50 gave 70.22% overall accuracy on the seen data (This performed well on unseen data)
		//75 gave 77.33%% overall accuracy on the seen data (The best found one for the unseen data)
		//100 gave 79.56% overall accuracy on the seen data (This performed well on unseen data)
		//150 gave 80.67% overall accuracy on the seen data (This performed well on unseen data)
		//300 gave 89.78% overall accuracy on the seen data (This caused over-fitting for unseen data)
		//Uses KD-Trees to perform nearest-neighbour lookup with the number of clusters - 100
		//Varying numbers of clusters will greatly affect the performance / speed - less clusters is less accurate for the seen data
		//However too many clusters means it will become too familiar with the seen data and perform poorly on unseen data.
		//Here it was found 150 gives a good accuracy on the unseen data as well as the seen data (80.67%)
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(75); //75 k-means 
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		//Perform clustering on the given data
		ByteCentroidsResult result = km.cluster(datasource);
		
		//Return the hard assigner
		return result.defaultHardAssigner();
	}
	
	//Record is a basic wrapper class which ties an FImage to it's ID, this is used to construct the MapBackedDataset, 
	//where the ID can act as the key.
	public static class Record implements Identifiable, ImageProvider<FImage> {

		private String id;
		private FImage image;
		
		//Construct the Record, give the object an id and image.
		Record(String id, FImage image) {
			this.id = id.split("\\.")[0];
			this.image = image;
		}

		//Call for the image
		@Override
		public FImage getImage() {
			return image;
		}

		//Call for the ID
		@Override
		public String getID() {
			return id;
		}
		
	}
	
	//Extractor is used to extract the features from the given data
	//This class is heavily influenced by the PHOWExtractor from the the OpenImaj tutorial 12, however instead a DenseSIFT is used instead of a pyramid dense sift.
	static class Extractor implements FeatureExtractor<DoubleFV, Record> {
	    DenseSIFT dsift = null;
	    HardAssigner<byte[], float[], IntFloatPair> assigner = null;
	    
	    //Assign the dense sift and assigner to the class variables
	    public Extractor(DenseSIFT dsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.dsift = dsift;
	        this.assigner = assigner;
	    }
	    
	    //Over-ridden method to extract the features for a given record
		@Override
		public DoubleFV extractFeature(Record object) {
			//Get the image for the record
			FImage image = object.getImage();

			//Analyse that image using the dense sift
	        dsift.analyseImage(image);
	    	
	        //Create a bag of visual words using the assigner
	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        //Perform spatial pooling on the local features
	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 2);

	        //Return the aggregated features into a vector, and then normalise it (to get a DoubleFV).
	        return spatial.aggregate(dsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
	}

}
