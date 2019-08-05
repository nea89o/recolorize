package recolorize.applications;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;


public class ColorizerRecordReader extends BaseImageRecordReader {

    public ColorizerRecordReader(final long height, final long width) {
        super(height, width, 3, (PathLabelGenerator) null);
    }

    private static double makeGrey(final double red, final double green, final double blue) {
        return 0.3 * red + 0.59 * green + 0.11 * blue;
    }

    @Override
    public List<Writable> next() {
        List<Writable> parentList = super.next();
        List<Writable> result = new ArrayList<>(parentList.size());

        for (Writable writable : parentList) {
            INDArray colorImageINDArray = ((NDArrayWritable) writable).get();

            INDArray greyImage = makeGreyImage(colorImageINDArray);
            INDArray ravelled = colorImageINDArray.ravel();

            result.add(new NDArrayWritable(greyImage));
            result.add(new NDArrayWritable(ravelled));
        }
        return result;
    }

    private INDArray makeGreyImage(final INDArray inputArray) {
        INDArray output = Nd4j.create(1, 1, height, width);

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                double red = inputArray.getDouble(0, 0, row, col);
                double green = inputArray.getDouble(0, 1, row, col);
                double blue = inputArray.getDouble(0, 2, row, col);

                double grey = makeGrey(red, green, blue);

                output.putScalar(0, 0, row, col, grey);
            }
        }
        return output;
    }
}