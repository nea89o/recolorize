package recolorize.image;


import java.awt.Color;

/**
 * Represents a part of an {@link Image}. It can be used as a proxy to access the original Image.
 *
 * @see Image
 */
public class ImageFrame {
    private final Image image;
    private final int x;
    private final int y;

    /**
     * Construct an {@link ImageFrame} based on an {@link Image}.
     *
     * @param image the image which is to proxy
     * @param x     the x coordinate of the center
     * @param y     the y coordinate of the center
     */
    public ImageFrame(Image image, int x, int y) {
        this.image = image;
        this.x = x;
        this.y = y;
    }

    /**
     * Get the color of a pixel relative to the center of this {@link ImageFrame}.
     *
     * @param offX the x offset
     * @param offY the y offset
     * @return the color of the pixel
     */
    public Color getPixelColor(int offX, int offY) {
        return image.getPixelColor(x + offX, y + offY);
    }

    /**
     * Get the color of the pixel relative to the center of this {@link ImageFrame}.
     *
     * @param offX the x offset
     * @param offY the y offset
     * @return the RGB color of the pixel
     */
    public int getPixel(int offX, int offY) {
        return image.getPixel(x + offX, y + offY);
    }

    /**
     * Gets the color of the center pixel.
     *
     * @return the RGB color of the center pixel
     */
    public int getPixel() {
        return getPixel(0, 0);
    }

    /**
     * Gets the color of the center pixel.
     *
     * @return the color of the center pixel
     */
    public Color getPixelColor() {
        return getPixelColor(0, 0);
    }
}
