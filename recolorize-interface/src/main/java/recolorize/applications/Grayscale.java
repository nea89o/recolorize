package recolorize.applications;

import javax.imageio.ImageIO;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Objects;

public class Grayscale {
    private static final String COLOR_IMAGES_DIR = "D:\\Daten\\Documents\\Colorization\\color";

    public static void main(String[] args) {
        for (File file : Objects.requireNonNull(new File(COLOR_IMAGES_DIR).listFiles())) {
            try {
                BufferedImage colorImage = ImageIO.read(file);
                BufferedImage grayImage = new BufferedImage(colorImage.getWidth(), colorImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
                Graphics g = grayImage.getGraphics();
                g.drawImage(colorImage, 0, 0, null);
                g.dispose();

                ImageIO.write(grayImage, "jpg", new File(file.getAbsolutePath().replace("color", "gray")));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
