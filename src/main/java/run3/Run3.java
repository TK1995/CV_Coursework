package run3;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

public class Run3 {

	private VFSGroupDataset<FImage> trainingSet;
	private VFSListDataset<FImage> testingSet;
	NaiveBayesAnnotator<Record, String> ann;
	
	GroupedRandomSplitter<String, Record> splits;
	GroupedDataset<String, ListDataset<Record>, Record> test;
	
	public static void main(String args[]) throws FileSystemException {
		Run3 run = new Run3();
		run.loadData();
		run.training();
		try {
			run.testing();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileSystemException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	// load the training and testing data set.
	public void loadData() throws FileSystemException {
		//this.trainingSet = new VFSGroupDataset<FImage>("/Users/tangke/Desktop/data/training",
		//		ImageUtilities.FIMAGE_READER);
		
		
		//this.testingSet = new VFSListDataset<FImage>("/Users/tangke/Desktop/data/testing",
		//		ImageUtilities.FIMAGE_READER);
		trainingSet = new VFSGroupDataset<FImage>("/Users/tangke/Desktop/data/training",
				ImageUtilities.FIMAGE_READER);
		//splits = new GroupedRandomSplitter<>(trainingSet, 70, 0, 30);
		
	}

	// Convert dataset from VFSGroupDataset to MapBackedDataset since the later one
	// has more functions
	public static MapBackedDataset<String, ListDataset<Record>, Record> convertDataset(VFSGroupDataset<FImage> dataset)
			throws FileSystemException {
		MapBackedDataset<String, ListDataset<Record>, Record> resultDataset = new MapBackedDataset<String, ListDataset<Record>, Record>();

		for (String cat : dataset.getGroups()) {
			ListDataset<FImage> imageSet = dataset.get(cat);
			FileObject[] fileObject = dataset.getFileObject(cat).getChildren();
			int catSize = imageSet.size();
			// create a new list which contains Records;
			ListDataset<Record> recordSet = new ListBackedDataset<Record>();

			for (int i = 0; i < catSize; i++) {
				Record newRecord = new Record(fileObject[i].getName().getBaseName(), imageSet.get(i));
				recordSet.add(newRecord);
			}

			resultDataset.put(cat, recordSet);
		}

		return resultDataset;

	}
	
	public void training() throws FileSystemException {
		
		// dsft with step size and bin size. 
		DenseSIFT dsft = new DenseSIFT(5, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsft, 6f, 7);
		
		MapBackedDataset<String, ListDataset<Record>, Record> trainSet = convertDataset(trainingSet);
		Dataset<Record> sampleSet = GroupedUniformRandomisedSampler.sample(trainSet, 30);
		
		splits = new GroupedRandomSplitter<>(trainSet, 70, 0, 30);
		test = splits.getTestDataset();
		
		System.out.println("Start to extract features.");
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(sampleSet, pdsift);
		
		//Create a kernelmap instance for the extractor
		HomogeneousKernelMap kernelmap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<DoubleFV, Record> extractor = kernelmap.createWrappedExtractor(new Extractor(assigner, pdsift));
		
		System.out.println("Start to train.");
		ann = new NaiveBayesAnnotator<Record, String>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
		//Start training the classifier
		ann.train(splits.getTrainingDataset());
		System.out.println("Train Finished.");
		
		
	}
	
	public void testing() throws FileNotFoundException, FileSystemException {
		System.out.println("Start to classify test set.");
		//FileObject[] fileObject = testingSet.getFileObjects()[0].getChildren();
		//FileObject[] fileObject = testingSet.getFileObjects();
		
		//PrintWriter pw = new PrintWriter("run3.txt");
		//String str;
		//for (int i = 0; i < testingSet.size(); i++) {
		//	Record record = new Record(fileObject[i].getName().getBaseName(), testingSet.get(i));
		//	str = fileObject[i].getName().getBaseName() + " "
		//			+ ann.classify(record).getPredictedClasses().iterator().next();
		//	pw.println(str);
		//}

		//pw.close();
		
		float correct = 0;
		float incorrect = 0;
		for (Entry<String, ListDataset<Record>> entry : splits.getTrainingDataset().entrySet()) {
			for (Record testImg : entry.getValue()) {
				String result = ann.classify(testImg).getPredictedClasses().iterator().next();
				if (result.equals(entry.getKey())) {
					correct++;
				} else
					incorrect++;
			}
		}
		float accuracy = correct / (correct + incorrect) * 100;

		System.out.println("Accuracy: " + accuracy);
	}

	// write HardAssigner based on SIFT features to perform K-means clustering
	// Inspired by OnpenImaj tutorial 12
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<Record> sampleSet, PyramidDenseSIFT<FImage> pdsift) {
		// Create the LocalFeatureList of ByteDSIFTKeypoint
		List<LocalFeatureList<ByteDSIFTKeypoint>> allKeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		for (Record record : sampleSet) {
			FImage image = (FImage) record.getImage();
			pdsift.analyseImage(image);
			// set the threadshold of the DSIFT features
			allKeys.add(pdsift.getByteKeypoints(0.005f));
		}

		// Start to compute k-means clustering
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(150);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allKeys);

		ByteCentroidsResult result = km.cluster(datasource);
		return result.defaultHardAssigner();

	}
	

	
	// write a extractor to extract features from given data
	static class Extractor implements FeatureExtractor<DoubleFV, Record> {
		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;

		public Extractor(HardAssigner<byte[], float[], IntFloatPair> assigner, PyramidDenseSIFT<FImage> pdsift) {
			this.assigner = assigner;
			this.pdsift = pdsift;
		}

		@Override
		public DoubleFV extractFeature(Record object) {
			FImage image = (FImage) object.getImage();

			pdsift.analyseImage(image);

			BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);
			BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw,
					2, 2);
			return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}

	}

}
