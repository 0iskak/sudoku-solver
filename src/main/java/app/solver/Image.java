package app.solver;

import java.io.IOException;

import static app.sudoku.ImageProc.overlay;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public class Image extends Solver {
    private final String url;

    public Image(String modelUrl, String url) {
        super(modelUrl);
        this.url = url;
        try {
            main();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void main() throws IOException {
        var image = imread(url);
        solve(image);
        try {
            show(overlay(image, numbers));
        } catch (NullPointerException ignored) {
            show(image);
        }
    }
}
