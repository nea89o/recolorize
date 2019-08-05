package recolorize.applications;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.scene.text.Text;
import recolorize.image.PixelImage;

import java.io.IOException;

public class MainInterfaceController {
    @FXML
    public Text textLul;
    @FXML
    public ImageView image;
    private int someNumber = 0;

    @FXML
    public void handleButton(ActionEvent actionEvent) throws IOException {
        image.setImage(PixelImage.load(MainInterfaceController.class.getResourceAsStream("/file.png")).asJavaFxImage());
    }
}
