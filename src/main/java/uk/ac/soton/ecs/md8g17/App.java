package uk.ac.soton.ecs.md8g17;

import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
    	/*
		 * VFSListDataset<FImage> scenes = new
		 * VFSListDataset<FImage>("/Users/tangke/Desktop/data/training",
		 * ImageUtilities.FIMAGE_READER); System.out.print(scenes.size());
		 * 
		 * DisplayUtilities.display("ATT Scenes", scenes);
		 */

		VFSGroupDataset<FImage> groupedScenes = new VFSGroupDataset<FImage>("/Users/tangke/Desktop/data/training",
				ImageUtilities.FIMAGE_READER);

		for (final Entry<String, VFSListDataset<FImage>> entry : groupedScenes.entrySet()) {
			DisplayUtilities.display(entry.getKey(), entry.getValue());
		}
        
    }
}
