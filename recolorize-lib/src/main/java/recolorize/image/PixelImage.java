package recolorize.image;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.function.Function;

public class PixelImage extends Image {

    private final int[] arr;
    private final int width;
    private final int height;

    private PixelImage(int[] arr, int width, int height) {
        this.arr = arr;
        this.width = width;
        this.height = height;
    }

    public static PixelImage fromJavaImage(java.awt.Image image) {
        return fromJavaImage(image, 0xffffff);
    }

    public static PixelImage fromJavaImage(java.awt.Image image, int backgroundColor) {
        BufferedImage bufferedImage = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
        bufferedImage.getGraphics().drawImage(image, 0, 0, null);
        return fromBufferedImage(bufferedImage, backgroundColor);
    }

    public static PixelImage fromBufferedImage(BufferedImage bufferedImage) {
        return fromBufferedImage(bufferedImage, 0xffffff);
    }

    public static PixelImage fromBufferedImage(BufferedImage bufferedImage, int backgroundColor) {
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        int[] buffer = new int[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int color = bufferedImage.getRGB(i, j);
                if ((color & 0xff000000) == 0) {
                    color = backgroundColor;
                }
                buffer[i * width + j] = color;
            }
        }
        return new PixelImage(buffer, width, height);
    }

    public static PixelImage load(InputStream stream) throws IOException {
        return load(stream, 0xffffff);
    }

    public static PixelImage load(InputStream stream, int backgroundColor) throws IOException {
        return fromBufferedImage(ImageIO.read(stream), backgroundColor);
    }

    public static PixelImage load(File file) throws IOException {
        return load(file, 0xffffff);
    }

    public static PixelImage load(File file, int backgroundColor) throws IOException {
        return fromBufferedImage(ImageIO.read(file), backgroundColor);
    }

    public static PixelImage fromPixelsArray(int[] arr, int width, int height) {
        return new PixelImage(arr, width, height);
    }

    @Override
    public Image mapFrame(Function<ImageFrame, Integer> mapper) {
        int[] buffer = new int[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                buffer[i * width + j] = mapper.apply(getImageFrameAround(i, j));
            }
        }
        return new PixelImage(buffer, width, height);
    }

    @Override
    public int getPixel(int x, int y) {
        if (x < 0 || x >= width) {
            throw new IllegalArgumentException("x must be in range 0-width");
        }
        if (y < 0 || y >= width) {
            throw new IllegalArgumentException("y must be in range 0-height");
        }
        return arr[x * width + y];
    }

    public int[] getPixels() {
        return arr;
    }

    @Override
    public BufferedImage asJavaImage() {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                image.setRGB(i, j, getPixel(i, j));
            }
        }
        return image;
    }

    @Override
    public File saveTo(File file) throws IOException {
        ImageIO.write(asJavaImage(), "PNG", file);
        return file;
    }
}
