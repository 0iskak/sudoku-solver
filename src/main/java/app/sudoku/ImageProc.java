package app.sudoku;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_ml.SVM;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import app.solver.Solver;

import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static app.solver.Solver.loader;

public class ImageProc {
    private static float[] lb, rb, lt, rt;

    public static Mat getGray(Mat image) {
        var gray = new Mat();

        cvtColor(image, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, new Size(3, 3), 0);

        return gray;
    }

    public static Mat getThreshold(Mat image) {
        var threshold = new Mat();

        adaptiveThreshold(image, threshold, 255,
                ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 8);

        return threshold;
    }

    public static Mat getBestContour(Mat threshold) {
        var contours = new MatVector();
        var maxArea = 0;
        Mat bestContour = null;

        findContours(threshold, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        for (Mat contour : contours.get()) {
            var area = (int) contourArea(contour);
            if (area > 1000 && area > maxArea) {
                maxArea = area;
                bestContour = contour;
            }
        }

        return bestContour;
    }

    public static Mat approximateContour(Mat contour) {
        var approx = new Mat();

        approxPolyDP(contour, approx, 0.1 * arcLength(contour, true), true);

        return approx;
    }

    public static Mat getProjected(Mat image, INDArray matrix) {
        lb = new float[]{matrix.getInt(0, 0, 2, 0), matrix.getInt(0, 1, 2, 0)};
        rb = new float[]{matrix.getInt(0, 0, 3, 0), matrix.getInt(0, 1, 3, 0)};
        lt = new float[]{matrix.getInt(0, 0, 1, 0), matrix.getInt(0, 1, 1, 0)};
        rt = new float[]{matrix.getInt(0, 0, 0, 0), matrix.getInt(0, 1, 0, 0)};

        var shape = new int[]{1, 4, 2};
        var pts1 = loader.asMat(Nd4j.create(new float[][]{rt, lt, lb, rb}).reshape(shape));
        var pts2 = loader.asMat(Nd4j.create(new float[][]{
                {Solver.WIDTH, 0},
                {0, 0},
                {0, Solver.HEIGHT},
                {Solver.HEIGHT, Solver.WIDTH}
        }).reshape(shape));

        var transform = getPerspectiveTransform(pts1, pts2);
        var result = new Mat();
        warpPerspective(image, result, transform, new Size(Solver.WIDTH, Solver.HEIGHT));

        return result;
    }

    public static void drawContour(Mat image, Mat contour) {
        drawContours(image, new MatVector(contour), 0, Scalar.GREEN);
    }

    public static byte[][] predictDigits(List<List<Mat>> digitImages,
                                         SVM svmModel, HOGDescriptor hog) {
        var grid = new byte[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                var digit = digitImages.get(i).get(j);
                if (digit != null)
                    grid[i][j] = predictDigit(digit, svmModel, hog);
            }
        }
        return grid;
    }

    private static byte predictDigit(Mat digit, SVM svmModel, HOGDescriptor hog) {
        var hist = Nd4j.create(hogCompute(digit, hog)).reshape(1, 1, -1);
        var result = svmModel.predict(loader.asMat(hist));
        return (byte) result;
    }

    private static float[] hogCompute(Mat digit, HOGDescriptor hog) {
        var winStride = new Size(1, 1);
        var padding = new Size(3, 3);
        var locations = new PointVector(new Point(8, 8));

        var descriptors = new float[1152];
        hog.compute(digit, descriptors, winStride, padding, locations);

        return descriptors;
    }

    public static void unProjectGrid(Mat outputImg, Mat image) {
        var shape = new int[]{1, 4, 2};
        var pts1 = loader.asMat(Nd4j.create(new float[][]{
                {Solver.WIDTH, 0},
                {0, 0},
                {0, Solver.HEIGHT},
                {Solver.WIDTH, Solver.HEIGHT}
        }).reshape(shape));
        var pts2 = loader.asMat(Nd4j.create(new float[][]{rt, lt, lb, rb}).reshape(shape));

        var transform = getPerspectiveTransform(pts1, pts2);
        warpPerspective(outputImg, outputImg, transform, image.size());
    }

    public static Mat overlay(Mat image, Mat overlay) throws IOException {
        var imageMatrix = loader.asMatrix(image);
        var overlayMatrix = loader.asMatrix(overlay);

        var shape = imageMatrix.shape();
        for (int i = 0; i < shape[2]; i++) {
            for (int j = 0; j < shape[3]; j++) {
                var i1 = new int[]{0, 0, i, j};
                var i2 = new int[]{0, 1, i, j};
                var i3 = new int[]{0, 2, i, j};

                var c1 = overlayMatrix.getInt(i1);
                var c2 = overlayMatrix.getInt(i2);
                var c3 = overlayMatrix.getInt(i3);

                var limit = 80;
                if (c3 < limit && c1 < limit)
                    continue;

                imageMatrix.putScalar(i1, c1);
                imageMatrix.putScalar(i2, c2);
                imageMatrix.putScalar(i3, c3);
            }
        }
        return loader.asMat(imageMatrix, CV_8UC3);
    }
}
