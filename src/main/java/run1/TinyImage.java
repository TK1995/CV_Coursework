package run1;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class TinyImage implements FeatureExtractor<FloatFV, FImage> {

	@Override
	public FloatFV extractFeature(FImage originalImage) {
		float[][] originalPixels = originalImage.pixels;

		int width = originalImage.getWidth();
		int height = originalImage.getHeight();
		int dimension = Math.min(width, height);
		float[][] pixels;

		// crop the image to a squre from the centre
		if (width > height) {
			pixels = new float[dimension][dimension];
			for (int x = 0; x < dimension; x++) {
				for (int y = 0; y < dimension; y++) {
					pixels[y][x] = originalPixels[y][(width - height) / 2 + x];
				}
			}
		} else {
			pixels = new float[dimension][dimension];
			for (int x = 0; x < dimension; x++) {
				for (int y = 0; y < dimension; y++) {
					pixels[y][x] = originalPixels[(height - width) / 2 + y][x];
				}
			}
		}
		FImage tinyImage = new FImage(pixels);
		tinyImage = ResizeProcessor.resample(tinyImage, 16, 16);
		
		float[][] tinyPixels = tinyImage.pixels;
		float sum = 0;
		for(int i=0; i<16; i++) {
			for(int j=0; j<16; j++) {
				sum += tinyPixels[i][j];
			}
		}
		
		// normalise the numbers in pixels
		float mean = sum / (16 * 16);
		float sd = 0;
		
		for(int i=0; i<16; i++) {
			for(int j=0; j<16; j++) {
				sd += (tinyPixels[i][j] - mean) * (tinyPixels[i][j] - mean);
			}
		}
		sd /= (16 * 16);
		
		for(int i=0; i<16; i++) {
			for(int j=0; j<16; j++) {
				tinyPixels[i][j] = (tinyPixels[i][j] - mean) / sd;
			}
		}
		
		// pack the image pixels into a vector
		float[] fv = new float[16*16];
		
		for(int i=0; i<16; i++) {
			for(int j=0; j<16; j++) {
				fv[i*16+j] = tinyPixels[i][j];
			}
		}
		
		FloatFV floatFV = new FloatFV(fv);

		return floatFV;
	}
	
	
}
