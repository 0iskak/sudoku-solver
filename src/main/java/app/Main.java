package app;

import app.solver.Image;
import com.formdev.flatlaf.FlatLightLaf;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.File;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        FlatLightLaf.setup();
        new Main();
    }

    private final JFileChooser fileChooser;
    private String xml;

    public Main() {
        fileChooser = new JFileChooser();
        getModel();
        getImage();
    }

    private void getImage() {
        while (true) {
            fileChooser.setDialogTitle("Choose sudoku to solve");
            fileChooser.setFileFilter(new FileNameExtensionFilter(
                    "Image files", getImageExtensions()));
            var state = fileChooser.showOpenDialog(null);
            if (state != JFileChooser.APPROVE_OPTION)
                break;

            new Image(xml, fileChooser.getSelectedFile().getAbsolutePath());
        }
    }

    private String[] getImageExtensions() {
        return Arrays.stream(ImageIO.getReaderFileSuffixes())
                .filter(Predicate.not(String::isBlank))
                .toArray(String[]::new);
    }

    private void getModel() {
        fileChooser.setDialogTitle("Choose model file");
        fileChooser.setFileFilter(new FileNameExtensionFilter(
                "XML files", "xml"));
        var state = fileChooser.showOpenDialog(null);
        if (state != JFileChooser.APPROVE_OPTION)
            getModel();

        xml = fileChooser.getSelectedFile().getAbsolutePath();
    }
}
