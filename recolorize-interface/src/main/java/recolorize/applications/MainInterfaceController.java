package recolorize.applications;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import recolorize.image.PixelImage;
import recolorize.neuralnetwork.ColorizerNeuralNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class MainInterfaceController {

    private MultiLayerNetwork model;

    private final int width = 256;
    private final int height = 256;

    private static final String USER_DIR = System.getProperty("user.home");
    private static final String RECOLORIZE_DIR = USER_DIR + "/recolorize";
    private static final String MODEL_NAME = "/model.zip";
    private static final String IMAGES_DIR = RECOLORIZE_DIR + "/images";

    @FXML
    public ImageView colorImageView;
    @FXML
    public ImageView greyImageView;

    public MainInterfaceController() throws IOException {
        new File(RECOLORIZE_DIR).mkdirs();
        new File(IMAGES_DIR).mkdirs();

        File netFile = new File(RECOLORIZE_DIR + MODEL_NAME);

        if (netFile.exists()) {
            InputStream netInputStream = new FileInputStream(new File(RECOLORIZE_DIR + MODEL_NAME));
            model = ModelSerializer.restoreMultiLayerNetwork(netInputStream);
        } else {
            ColorizerNeuralNetwork net = new ColorizerNeuralNetwork(width, height);
            net.loadImages(IMAGES_DIR);
            model = net.train(5);

            ModelSerializer.writeModel(model, RECOLORIZE_DIR + MODEL_NAME, true);
        }
    }

    @FXML
    public void load() throws IOException {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Load image");
        File imageFile = fileChooser.showOpenDialog(null);

        if (imageFile != null && model != null) {
            PixelImage greyImage = PixelImage.load(imageFile);
            PixelImage resizedGreyImage = greyImage.resize(width, height);
            greyImageView.setImage(resizedGreyImage.resize(greyImage.getWidth(), greyImage.getHeight()).asJavaFxImage());

            INDArray array = Nd4j.create(new int[]{1, width, height}, intArrayToDoubleArray(resizedGreyImage.getPixels()));

            int[] colorImagePixels = model.output(array).toIntVector();

            PixelImage colorImage = PixelImage.fromPixelsArray(colorImagePixels, width, height);
            colorImageView.setImage(colorImage.resize(greyImage.getWidth(), greyImage.getHeight()).asJavaFxImage());
        }
    }

    private double[] intArrayToDoubleArray(int[] arr) {
        return Arrays.stream(arr).asDoubleStream().toArray();
    }
}
