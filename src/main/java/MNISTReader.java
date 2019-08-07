import org.bytedeco.tesseract.ROW;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MNISTReader {

    public static final int LABEL_FILE_OFFSET = 8;
    public static final int IMAGE_FILE_OFFSET = 16;
    public static final int ROW_NUMBER = 28;
    public static final int COLUMN_NUMBER = 28;
    public static int trainDataSetLength;
    public static int testDataSetLength;

    private static FileInputStream readImages(File file) {
        FileInputStream in = null;
        try {
            in = new FileInputStream(file);

            in.skip(IMAGE_FILE_OFFSET);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return in;
    }


    public static List<int[][]> readImagesMatrix(File file) throws IOException {
        ArrayList<int[][]> images = null;
        images = new ArrayList<int[][]>();
        FileInputStream in = readImages(file);
        int pixels = ROW_NUMBER*COLUMN_NUMBER;
        byte[] imageBytes = new byte[pixels];

        int[][] image = new int[ROW_NUMBER][COLUMN_NUMBER];

        while (in.read(imageBytes, 0, pixels) != -1) {

            for (int i = 0; i < ROW_NUMBER; i++) {
                for (int j = 0; j < COLUMN_NUMBER; j++) {
                    image[i][j] = imageBytes[i * ROW_NUMBER + j] & 0xFF;
                }
            }
            images.add(image);
            image = new int[ROW_NUMBER][COLUMN_NUMBER];
        }
        return images;
    }


    public static double[][] readImagesVector(File file) throws IOException {
        ArrayList<double[]> images = null;
            FileInputStream in = readImages(file);
            int pixels = ROW_NUMBER*COLUMN_NUMBER;
            byte[] imageBytes = new byte[pixels];

            double[] image = new double[ROW_NUMBER*COLUMN_NUMBER];
            int n = image.length;

            images = new ArrayList<double[]>();
            while(in.read(imageBytes, 0, pixels)!=-1) {

                for(int i=0; i<n; i++) {
                    image[i] = imageBytes[i] & 0xFF;
                }
                images.add(image);
                image = new double[ROW_NUMBER*COLUMN_NUMBER];
            }
            double[][] imagesVec = new double[images.size()][];
            for(int i=0; i<imagesVec.length; i++) {
                imagesVec[i] = images.get(i);
            }

        return imagesVec;
    }

    public static double[] readLabels(File file) throws IOException {

        double[] labels = null;
        List<Double> labelsList = new ArrayList<Double>();
        FileInputStream in = new FileInputStream(file);

        try {

            in.skip(LABEL_FILE_OFFSET);
            int label;
            while((label=in.read()) != -1) {
                labelsList.add(Double.valueOf(label));
            }
        } catch (FileNotFoundException e) {
            e.fillInStackTrace();
        }

        labels = new double[labelsList.size()];
        for(int i=0; i<labels.length; i++) {
            labels[i] = labelsList.get(i);
        }

        return labels;
    }

}
