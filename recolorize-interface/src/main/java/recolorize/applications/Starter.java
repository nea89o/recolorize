package recolorize.applications;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

import java.io.IOException;

public class Starter extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws IOException {
        Parent root = FXMLLoader.load(MainInterfaceController.class.getResource("/main_interface.fxml"));

        Scene scene = new Scene(root, 800, 600);
        stage.setTitle("Colorization");
        stage.setScene(scene);
        stage.getIcons().add(new Image(getClass().getResourceAsStream("/icon.png")));
        stage.setMaximized(true);
        stage.show();
    }
}
