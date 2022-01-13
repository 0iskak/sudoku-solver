package app.sudoku;

import de.sfuhrm.sudoku.GameMatrixFactory;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import app.solver.Solver;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static app.solver.Solver.loader;

public class SudokuSolver {
    public static HOGDescriptor getHogDescriptor() {
        var winSize = new Size(18, 18);
        var blockSize = new Size(3, 4);
        var blockStride = new Size(1, 2);
        var cellSize = new Size(3, 4);
        var nBins = 9;
        var deriveAperture = 20;
        var winSigma = 4d;
        var histogramNormType = 0;
        var L2HysThreshold = 2d;
        var gammaCorrection = false;
        var nLevels = 64;
        var signedGradient = false;

        return new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins, deriveAperture,
                winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nLevels, signedGradient);
    }

    public static byte[][] solveSudoku(byte[][] grid) {
        var matrix = new GameMatrixFactory().newGameMatrix();
        matrix.setAll(grid);

        var solver = new de.sfuhrm.sudoku.Solver(matrix);
        solver.setLimit(1);

        return solver.solve().get(0).getArray();
    }

    public static void drawOriginalNumbers(Mat image, byte[][] grid) {
        var numWidth = Solver.WIDTH / 9;
        var padding = 10;

        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (grid[i][j] != 0)
                    putText(image, String.valueOf(grid[i][j]),
                            new Point(j * numWidth + padding + 25,
                                    i * numWidth + padding + 10 + numWidth / 2),
                            FONT_HERSHEY_DUPLEX, 0.7, new Scalar(0, 0, 255, 255),
                            2, LINE_8, false);
    }

    public static void drawSolvedNumbers(Mat image, byte[][] gridOld, byte[][] grid) {
        var numWidth = Solver.WIDTH / 9;
        var padding = 10;

        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (gridOld[i][j] == 0)
                    putText(image, String.valueOf(grid[i][j]),
                            new Point(j * numWidth + padding,
                                    i * numWidth + padding + numWidth / 2),
                            FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 0, 255),
                            2, LINE_8, false);
    }

    public static List<List<Mat>> getDigits(Mat image) throws IOException {
        var margin = 9;
        var blockSize = Solver.WIDTH / 9;

        var digitImages = new LinkedList<List<Mat>>();

        for (int y = 0; y < Solver.HEIGHT; y += blockSize) {
            var row = new LinkedList<Mat>();
            for (int x = 0; x < Solver.WIDTH; x += blockSize) {
                var rect = new Rect(x + margin, y + margin,
                        blockSize - margin, blockSize - margin);
                var digit = new Mat(image, rect);

                row.add(isEmpty(digit) ? null : digit);
            }

            digitImages.add(row);
        }

        return digitImages;
    }

    public static boolean isEmpty(Mat image) throws IOException {
        var matrix = loader.asMatrix(image);
        return matrix.meanNumber().intValue() < matrix.minNumber().intValue() + 20;
    }
}
