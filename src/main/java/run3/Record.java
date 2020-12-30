package run3;

import org.openimaj.data.identity.Identifiable;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageProvider;

/* This class is used to get image tied with its own id.
 * The id is later used as key.
 * */
public class Record implements  Identifiable, ImageProvider{
	private String id;
	private FImage image;
	
	//Get rid of .jpg for the id.
	Record(String id, FImage image){
		this.id = id.split("\\.")[0];
		this.image = image;
	}
	
	@Override
	public String getID() {
		// TODO Auto-generated method stub
		return id;
	}
	@Override
	public Image getImage() {
		// TODO Auto-generated method stub
		return image;
	}

}
