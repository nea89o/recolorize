package recolorize.applications;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import recolorize.neuralnetwork.ColorizerNeuralNetwork;

import java.io.IOException;

import static recolorize.applications.Constants.*;


public class ColorizerNeuralNetworkTrainer {

    public static void main(String[] args) throws IOException {
        ColorizerNeuralNetwork net = new ColorizerNeuralNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGES_DIR);
        MultiLayerNetwork model = net.train(1);

        ModelSerializer.writeModel(model, RECOLORIZE_DIR + MODEL_NAME, true);
    }
}
