package recolorize.applications;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import recolorize.image.PixelImage;

import java.io.File;
import java.io.IOException;

import static recolorize.applications.Colorizer.HEIGHT;
import static recolorize.applications.Colorizer.WIDTH;

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

            INDArray array = Nd4j.create(intArrayToDoubleArray(greyImage.getPixels()));
            INDArray resized = resizeNDArray(array, new int[]{HEIGHT, WIDTH});

            int[] colorImagePixels = model.output(resized).toIntVector();

            PixelImage colorImage = PixelImage.fromPixelsArray(colorImagePixels, WIDTH, HEIGHT);
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

    private INDArray resizeNDArray(INDArray arr, int [] shape) {
        INDArray resized = Nd4j.create(shape);
        resized.get(NDArrayIndex.createCoveringShape(arr.shape())).assign(arr);
        return resized;
    }
}
