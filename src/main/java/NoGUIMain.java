import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NoGUIMain {

    public static void main(String[] args) throws IOException {
        System.out.println("Welcome in ND4J neural network implementation program. To choose option type letter in brackets and press enter.");

        System.out.println("Would you like to train and test [a]ll neural networks or choose [o]ne configration?");
        Scanner scanner = new Scanner(System.in);
        String option = scanner.next();
        Controller controller = new Controller();
        controller.readDataSet();

        if(option.equals("a")) {

            System.out.println("Starting...");

            StringBuilder sb = new StringBuilder();
            sb.append("TYPE,P_TEST,A_TEST,P_TRAIN,A_TRAIN\n");

            for (NNConfBuilder.NNConf C : NNConfBuilder.NNConf.values()) {
                System.out.println("Training: " + C.toString());
                controller.trainNeuralNetwork(C, 200);
                System.out.println("Testing: " + C.toString());
                double[] metrics = controller.testNeuralNetwork();
                String result = "P: " + metrics[0] + "\tA: " + metrics[1];
                System.out.println(result);
                sb.append(C.toString() + "," + metrics[2] + "," + metrics[3] + "," + metrics[0] + "," + metrics[1] + "\n");
            }

            FileWriter fw = new FileWriter("results.csv");
            fw.write(sb.toString());
            fw.close();
            System.out.println("Success, results saved in results.csv");
        } else if(option.equals("o")) {
            System.out.println("To choose neural net configuration type number and press ENTER: ");
            StringBuilder sb_nn_config = new StringBuilder();
            for(NNConfBuilder.NNConf C: NNConfBuilder.NNConf.values()) {
                sb_nn_config.append(C.ordinal() + ". Neurons in hiden layers: " + Arrays.toString(C.neurons));
                for(int i=0; i<4-C.hiddenLayers; i++) {
                    sb_nn_config.append("\t");
                }
                sb_nn_config.append("Activation: " +  C.activation.toString() + "\n");
            }
            System.out.println(sb_nn_config.toString());
            int nnOption = scanner.nextInt();
            System.out.println("Choose number of epochs: ");
            int epochs = scanner.nextInt();
            System.out.println("Training...");
            controller.trainNeuralNetwork(NNConfBuilder.NNConf.values()[nnOption], epochs);
            System.out.println("Testing...");
            System.out.println(controller.getFullReport());



        }
    }
}
