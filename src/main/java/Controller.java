import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.IOException;

public class Controller {

    EmnistDataSetIterator testDataSet;
    EmnistDataSetIterator trainDataSet;
    EmnistDataSetIterator.Set emnistSet;
    MultiLayerNetwork net;

    public int readDataSet() {

        int batchSize = 16;

        emnistSet = EmnistDataSetIterator.Set.MNIST;
        try {
            trainDataSet = new EmnistDataSetIterator(emnistSet, batchSize, true);
            testDataSet = new EmnistDataSetIterator(emnistSet, batchSize, false);
        } catch (IOException e) {
            e.printStackTrace();
            return 1;
        }
        return 0;
    }


//    public void readTestSet(String labelsPath, String dataPath) {
//        this.testDataSet = this.readDataSet(labelsPath, dataPath);
//    }
//
//    public void readTrainSet(String labelsPath, String dataPath) {
//        this.trainDataSet = this.readDataSet(labelsPath, dataPath);
//    }

//    public void printTestData() {
////        for(int i=0; i<10; i++) {
////            System.out.println(this.testDataSet.getFeatures().getRow(i));
//            System.out.println(this.testDataSet.getFeatures().getRow(0L).reshape(28L, 28L));
////        }
//    }

//    public void printTestLabels() {
//        for(int i=0; i<10; i++) {
//            System.out.println(this.);
//        }
//    }

    public void trainNeuralNetwork(NNConfBuilder.NNConf configOption, int epochs) {
        MultiLayerConfiguration config = new NNConfBuilder(emnistSet).build(configOption);
        net = new MultiLayerNetwork(config);
        net.pretrain(trainDataSet);
        net.fit(trainDataSet);
    }

    public double[] testNeuralNetwork() {
        Evaluation eval_test = net.evaluate(testDataSet);
        Evaluation eval_train = net.evaluate(trainDataSet);
        return new double[]{eval_train.precision(), eval_train.accuracy(), eval_test.precision(), eval_test.accuracy()};

    }

    public String getFullReport() {
        return net.evaluate(testDataSet).stats();
    }

    public void loadNNetworkConfig(int index) {

    }


}
