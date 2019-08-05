package recolorize.image;

import javafx.embed.swing.SwingFXUtils;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.function.Function;

/**
 * Interface to represent an image.
 */
public abstract class Image {
    /**
     * Transform the image pixel by pixel.
     *
     * @param mapper a function which maps a pixel {@link Color} to a new one.
     *
     * @return a new {@link Image}
     */
    public Image mapPixelColors(Function<Color, Color> mapper) {
        return mapPixels(i -> mapper.apply(new Color(i)).getRGB());
    }

    /**
     * Transform the image pixel by pixel.
     *
     * @param mapper a function which maps a pixel to a new one.
     *
     * @return a new {@link Image}
     */
    public Image mapPixels(Function<Integer, Integer> mapper) {
        return mapFrame(frame -> mapper.apply(frame.getPixel()));
    }

    /**
     * Transforms the image pixel by pixel. Each step has a {@link ImageFrame} given to represent the context of the
     * transforming pixel.
     *
     * @param mapper a function which maps a pixel {@link Color} with context to a new one
     *
     * @return a new {@link Image}
     */
    public Image mapFrameToColor(Function<ImageFrame, Color> mapper) {
        return mapFrame(frame -> mapper.apply(frame).getRGB());
    }

    /**
     * Transforms the image pixel by pixel. Each step has a {@link ImageFrame} given to represent the context of the
     * transforming pixel.
     *
     * @param mapper a function which maps a pixel with context to a new one
     *
     * @return a new {@link Image}
     */
    public abstract Image mapFrame(Function<ImageFrame, Integer> mapper);

    /**
     * Gets a pixel {@link Color}
     *
     * @param x the x coordinate of the pixel
     * @param y the y coordinate of the pixel
     *
     * @return the {@link Color} of the pixel
     */
    public Color getPixelColor(int x, int y) {
        return new Color(getPixel(x, y));
    }

    /**
     * Gets a pixel {@link Color}
     *
     * @param x the x coordinate of the pixel
     * @param y the y coordinate of the pixel
     *
     * @return the {@link Color} of the pixel
     */
    public abstract int getPixel(int x, int y);

    /**
     * Convert this image to a java {@link BufferedImage}
     *
     * @return this image as a java image
     */
    public abstract BufferedImage asJavaImage();

    /**
     * Saves the Image to a file. The supplied file should have an {@code .PNG} file ending
     *
     * @param file the file the image should be saved to
     *
     * @return the location of the file
     */
    public abstract File saveTo(File file) throws IOException;

    /**
     * Get an {@link ImageFrame} of this image centered around (x, y)
     *
     * @param x the x coordinate of the center
     * @param y the y coordinate of the center
     *
     * @return the {@link ImageFrame}
     */
    public ImageFrame getImageFrameAround(int x, int y) {
        return new ImageFrame(this, x, y);
    }

    public javafx.scene.image.Image asJavaFxImage() {
        return SwingFXUtils.toFXImage(asJavaImage(), null);
    }
}
