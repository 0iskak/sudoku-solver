package app.solver;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_ml.SVM;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

import static app.sudoku.ImageProc.*;
import static app.sudoku.SudokuSolver.*;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;

abstract public class Solver {
    public static int WIDTH = 450;
    public static int HEIGHT = 450;
    public static NativeImageLoader loader = new NativeImageLoader();

    protected final HOGDescriptor hog;
    protected final SVM svmModel;
    protected byte[][] grid, gridOld;
    protected INDArray matrix;
    protected Mat numbers;

    protected Solver(String modelUrl) {
        hog = getHogDescriptor();
        svmModel = SVM.load(modelUrl);
    }

    protected void solve(Mat image) throws IOException {
        var gray = getGray(image);
        var threshold = getThreshold(gray);
        var bestContour = getBestContour(threshold);
        if (bestContour == null) return;

        var approx = approximateContour(bestContour);
        if (approx.rows() != 4) return;

        matrix = loader.asMatrix(approx);
        var projectedGrid = getProjected(threshold, matrix);
        var digitImages = getDigits(projectedGrid);

        grid = predictDigits(digitImages, svmModel, hog);
        gridOld = new byte[9][9];
        System.arraycopy(grid, 0, gridOld, 0, 9);
        grid = solveSudoku(grid);

        numbers = new Mat(image.size(), CV_8UC3, Scalar.BLACK);
        drawOriginalNumbers(numbers, gridOld);
        drawSolvedNumbers(numbers, gridOld, grid);
        unProjectGrid(numbers, image);
    }

    protected void show(Mat image) {
        imshow("Sudoku", image);
        waitKey(0);
    }
}
