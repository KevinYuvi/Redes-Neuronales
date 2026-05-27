import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;

public class Main {

    public static void main(String[] args) throws Exception {

        int batchSize = 64;
        int epochs = 3;
        int seed = 123;

        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, seed);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        long startTrain = System.nanoTime();

        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
        }

        long endTrain = System.nanoTime();

        long startEval = System.nanoTime();
        var evaluation = model.evaluate(testData);
        long endEval = System.nanoTime();

        double trainSeconds = (endTrain - startTrain) / 1_000_000_000.0;
        double evalSeconds = (endEval - startEval) / 1_000_000_000.0;

        ModelSerializer.writeModel(
                model,
                new File("mnist_model_java.zip"),
                true
        );

        System.out.println("Modelo Java guardado");

        System.out.println("RESULTADOS JAVA - Deeplearning4j");
        System.out.println("--------------------------------");
        System.out.println("Accuracy: " + evaluation.accuracy());
        System.out.println("Tiempo entrenamiento: " + trainSeconds + " segundos");
        System.out.println("Tiempo inferencia/evaluacion: " + evalSeconds + " segundos");
    }
}