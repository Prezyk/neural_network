import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.rmi.activation.ActivationID;
import java.util.Arrays;

public class NNConfBuilder {


    public NNConfBuilder(EmnistDataSetIterator.Set dataSet) {
        this.dataSet = dataSet;
    }


    private int seed = 123;
    private int rowsNum = 28;
    private int colNum = 28;
    private EmnistDataSetIterator.Set dataSet;


    public MultiLayerConfiguration build(NNConf confOption) {

        int[] totalNeurons = new int[confOption.hiddenLayers+2];
        int lastLayerIndex = totalNeurons.length-1;
        for(int i=0; i<totalNeurons.length; i++) {
            if(i==0) {
                totalNeurons[i] = rowsNum*colNum;
            } else if(i == lastLayerIndex) {
                totalNeurons[i] = EmnistDataSetIterator.numLabels(dataSet);
            } else {
                totalNeurons[i] = confOption.neurons[i-1];
            }
        }



        NeuralNetConfiguration.ListBuilder config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .l2(0.0001)
                .list();

//        //input layer
//        config = config.layer(0, new DenseLayer.Builder()
//        .nIn(colNum*rowsNum)
//        .nOut(confOption.neurons[0])
//        .activation(confOption.activation)
//        .weightInit(WeightInit.XAVIER).build());

        //output layer


        int layerIndex = 0;
        do {
            config = config.layer(layerIndex, new DenseLayer.Builder()
            .nIn(totalNeurons[layerIndex])
            .nOut(totalNeurons[layerIndex+1])
            .activation(confOption.activation)
            .weightInit(WeightInit.XAVIER)
            .build());

            ++layerIndex;

        } while(layerIndex<lastLayerIndex-1);


        config = config.layer(lastLayerIndex-1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(totalNeurons[lastLayerIndex-1])
                .nOut(totalNeurons[lastLayerIndex])
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .build());

        return config.build();
    }







    public enum NNConf {

        C_1H_5N_SIGM(1, new int[]{5}, Activation.SIGMOID),
        C_1H_10N_SIGM(1, new int[]{10}, Activation.SIGMOID),
        C_1H_30N_SIGM(1, new int[]{30}, Activation.SIGMOID),
        C_1H_50N_SIGM(1, new int[]{50}, Activation.SIGMOID),
        C_2H_30N_SIGM(2, new int[]{15, 15}, Activation.SIGMOID),
        C_3H_30N_SIGM(3, new int[]{10, 10, 10}, Activation.SIGMOID),
        C_2H_60N_SIGM(2, new int[]{30, 30}, Activation.SIGMOID),
        C_3H_90N_SIGM(3, new int[]{30, 30, 30}, Activation.SIGMOID),
        C_1H_30N_RELU(1, new int[]{30}, Activation.RELU),
        C_1H_30N_TANH(1, new int[]{30}, Activation.TANH);


        public int hiddenLayers;
        public int[] neurons;
        public Activation activation;

        NNConf(int hiddenLayers, int[] neurons, Activation activation) {
            this.hiddenLayers = hiddenLayers;
            this.neurons = neurons;
            this.activation = activation;
        }

    }

}

