package recolorize.applications;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class Colorizer {
    private static final Logger LOG = LoggerFactory.getLogger(Colorizer.class);

    static final int HEIGHT = 32;
    static final int WIDTH = 32;
    private static final int CHANNELS = 1;
    private static final int BATCH_SIZE = 32;
    private static final int SEED = 123;
    private static final int EPOCHS = 5;
    private static final String MODEL_PATH = "recolorize-interface/src/main/resources/model.zip";
    private static final String IMAGES_DIRECTORY = "D:\\Daten\\Documents\\Colorization\\color";
    private static final String ALLOWED_FORMATS = "jpg";

    public static void main(final String[] args) throws IOException {
        LOG.info("LOAD DATA");
        File imagesDirectory = new File(IMAGES_DIRECTORY);

        FileSplit images = new FileSplit(imagesDirectory, new String[]{ALLOWED_FORMATS});

        ColorizerRecordReader recordReader = new ColorizerRecordReader(HEIGHT, WIDTH);
        recordReader.initialize(images);

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE);

        LOG.info("CREATE MODEL");
        MultiLayerNetwork net = lenet();
        net.init();

        LOG.info("TRAIN MODEL");
        for (int i = 0; i < EPOCHS; i++) {
            net.fit(iterator);
            LOG.info("FINISHED EPOCH " + i);
        }

        LOG.info("SAVE MODEL");
        ModelSerializer.writeModel(net, MODEL_PATH, false);
    }

    private static MultiLayerNetwork lenet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.0001, 0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .nOut(50)
                        .biasInit(0.0)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn2")
                        .nOut(100)
                        .biasInit(0)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(HEIGHT * WIDTH)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
                .build();
        return new MultiLayerNetwork(conf);
    }

    private static MultiLayerNetwork alexnet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new AdaDelta())
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(5 * 1e-4)
                .list()
                .layer(new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(1)
                        .nOut(96)
                        .biasInit(0)
                        .build())
                .layer(new LocalResponseNormalization.Builder()
                        .name("lrn1")
                        .build())
                .layer(new SubsamplingLayer.Builder(new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(1)
                        .build())
                .layer(new LocalResponseNormalization.Builder()
                        .name("lrn2")
                        .build())
                .layer(new SubsamplingLayer.Builder(new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(384)
                        .biasInit(0)
                        .build())
                .layer(new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(1)
                        .build())
                .layer(new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(256)
                        .biasInit(1)
                        .build())
                .layer(new SubsamplingLayer.Builder(new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool3")
                        .build())
                .layer(new DenseLayer.Builder()
                        .name("ffn1").nOut(4096)
                        .biasInit(1)
                        .dropOut(0.5)
                        .build())
                .layer(new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .biasInit(1)
                        .dropOut(0.5)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(HEIGHT * WIDTH)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
                .build();
        return new MultiLayerNetwork(conf);
    }
}
