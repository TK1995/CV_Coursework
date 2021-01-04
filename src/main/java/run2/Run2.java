package run2;

import java.io.IOException;
import java.io.FileNotFoundException;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.vfs2.FileObject;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

import FileSorter.FileSorter;

public class Run2 {

	public static int patchSize = 8;
	public static int step = 4;
	private LiblinearAnnotator<FImage, String> ann;

	public void train(VFSGroupDataset<FImage> trainSet) {
		GroupedRandomSplitter<String, FImage> splitSet = new GroupedRandomSplitter<String, FImage>(trainSet, 15, 0, 0);

		HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(splitSet.getTrainingDataset());
		// System.out.println("Assigner completed.");
		FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(assigner, patchSize, step);
		System.out.println("Features Extration Completed.");

		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(trainSet);
		System.out.println("Training Finished.");
	}

	public ClassificationResult<String> classify(FImage image) {
		return ann.classify(image);
	}

	/*
	 * Build a HardAssigner based on k-means, to be run on the patches sampled from image datasets
	 */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> trainSet) {

		// this list is used to store all the vectors from all the images.
		List<float[]> vocabulary = new ArrayList<>();

		for (FImage image : trainSet) {
			// Get the feature vectors of all images
			// System.out.println("img.................");
			List<LocalFeature<SpatialLocation, FloatFV>> fvs = extractor(image, patchSize, step);
			// System.out.println("fvs.................");
			for (LocalFeature<SpatialLocation, FloatFV> fv : fvs) {
				vocabulary.add(fv.getFeatureVector().values);
				// System.out.println("fv------------------");
			}
		}

		if (vocabulary.size() > 100000) {
			Collections.shuffle(vocabulary);
			vocabulary = vocabulary.subList(0, 100000);
		}
		System.out.println("Sample size is: " + vocabulary.size());

		float[][] data = new float[vocabulary.size()][];
		for (int i = 0; i < vocabulary.size(); i++) {
			data[i] = vocabulary.get(i);
		}
		System.out.println("One vector size is: " + data.length);

		System.out.println("----------------Start clustering!------------------");
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(300);
		// DataSource<float[]> datasource = new LocalFeatureListDataSource<>(vocabulary);
		FloatCentroidsResult output = km.cluster(data);
		System.out.println("assigner finished.................");
		return output.defaultHardAssigner();
	}

	// Conveniently obtain the feature vectors and their location, using the LocalFeature class
	public static List<LocalFeature<SpatialLocation, FloatFV>> extractor(FImage image, int patchSize, int step) {

		List<LocalFeature<SpatialLocation, FloatFV>> result = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

		// A simple mechanism to create a sliding window over an image, retrieving all positions:
		RectangleSampler sampler = new RectangleSampler(image, step, step, patchSize, patchSize);
		List<Rectangle> allPositions = sampler.allRectangles();

		for (Rectangle p : allPositions) {
			FImage patch = image.extractROI(p);
			FImage normalizedPatch = patch.normalise();
			float[][] patchPixels = normalizedPatch.pixels;
			float[] vector = ArrayUtils.reshape(patchPixels);

			// use the 1d vector above to generate a feature vector
			FloatFV featureVector = new FloatFV(vector);
			// SpatialLocation location = new SpatialLocation(patch.height, patch.width);
			LocalFeature<SpatialLocation, FloatFV> x = new LocalFeatureImpl<SpatialLocation, FloatFV>(
					new SpatialLocation(p.x, p.y), featureVector);
			result.add(x);
		}
		return result;
	}

	class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage> {

		HardAssigner<float[], float[], IntFloatPair> assigner;
		int patchSize, step;

		public BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, int patchSize, int step) {
			this.assigner = assigner;
			this.patchSize = patchSize;
			this.step = step;
		};

		@Override
		public DoubleFV extractFeature(FImage object) {
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
			return spatial.aggregate(extractor(object, patchSize, step), object.getBounds()).normaliseFV();
		}
	}

	public static void main(String[] args) throws FileSystemException, FileNotFoundException, IOException {

		VFSGroupDataset<FImage> trainScenes = new VFSGroupDataset<FImage>("/Users/tangke/Desktop/data/training",ImageUtilities.FIMAGE_READER);
		VFSListDataset<FImage> testScenes = new VFSListDataset<>("/Users/tangke/Desktop/data/testing",ImageUtilities.FIMAGE_READER);

		Run2 app = new Run2();
		app.train(trainScenes);

		System.out.println("Start to classify test set.");
		FileObject[] files = testScenes.getFileObjects();

		PrintWriter pw = new PrintWriter("run2.txt");
		String str;
		for (int i = 0; i < testScenes.size(); i++) {
			str = files[i].getName().getBaseName() + " "
					+ app.classify(testScenes.get(i)).getPredictedClasses().iterator().next();
			pw.println(str);
		}
		pw.close();
		
		// sort output file lines in numerical order
		FileSorter fs = new FileSorter();
		fs.sortFile("run2.txt");
	}
}