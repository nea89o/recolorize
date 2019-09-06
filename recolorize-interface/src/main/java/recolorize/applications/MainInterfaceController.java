package recolorize.applications;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.factory.Nd4j;
import recolorize.image.PixelImage;

import java.io.File;
import java.io.IOException;

public class MainInterfaceController {

  private final MultiLayerNetwork model;

  @FXML
  public ImageView colorImageView;
  @FXML
  public ImageView greyImageView;

  public MainInterfaceController() throws IOException {
    model = ModelSerializer.restoreMultiLayerNetwork(getClass().getResourceAsStream("/model.zip"));
  }

  @FXML
  public void load() throws IOException {
    FileChooser fileChooser = new FileChooser();
    fileChooser.setTitle("Load image");
    File imageFile = fileChooser.showOpenDialog(null);

    if (imageFile != null) {
      PixelImage greyImage = PixelImage.load(imageFile);
      greyImageView.setImage(greyImage.asJavaFxImage());

      int[] greyImagePixels = model.predict(Nd4j.create(intArrayToDoubleArray(greyImage.getPixels())));

      PixelImage colorImage = PixelImage.fromPixelsArray(greyImagePixels, Colorizer.WIDTH, Colorizer.HEIGHT);
      colorImageView.setImage(colorImage.asJavaFxImage());
    }
  }

  private double[] intArrayToDoubleArray(int[] arr) {
    double[] doubleArray = new double[arr.length];
    for (int i = 0; i < arr.length; i++) {
      doubleArray[i] = arr[i];
    }
    return doubleArray;
  }
}
