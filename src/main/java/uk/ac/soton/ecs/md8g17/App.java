package uk.ac.soton.ecs.md8g17;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
    	
        VFSGroupDataset<FImage> groupedScenes = new VFSGroupDataset<FImage>(
        	    "/Users/maggie/Downloads/training", ImageUtilities.FIMAGE_READER);
        
        System.out.println(groupedScenes.size());
        
        FImage randomImage = groupedScenes.getRandomInstance();
        DisplayUtilities.display(randomImage);
        
    }
}
