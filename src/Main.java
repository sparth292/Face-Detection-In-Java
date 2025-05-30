import javax.swing.*;
import java.awt.image.BufferedImage;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfRect;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        System.out.println("OpenCV loaded successfully. Version: " + Core.VERSION);

        JFrame frame = new JFrame("Face Detection");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel imageLabel = new JLabel();
        frame.add(imageLabel);
        frame.setSize(640, 480);
        frame.setVisible(true);

        // Load face cascade
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        if (faceDetector.empty()) {
            System.out.println(" Failed to load cascade classifier!");
            return;
        } else {
            System.out.println(" Cascade loaded successfully.");
        }

        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.out.println(" Cannot open webcam.");
            return;
        }

        new Thread(() -> {
            Mat frameMat = new Mat();
            Mat grayMat = new Mat();
            MatOfRect faces = new MatOfRect();

            while (true) {
                if (!capture.read(frameMat)) continue;

                // Convert to grayscale
                Imgproc.cvtColor(frameMat, grayMat, Imgproc.COLOR_BGR2GRAY);

                // Improve contrast using histogram equalization
                Imgproc.equalizeHist(grayMat, grayMat);

                // Optional: Reduce noise slightly to help detection
                Imgproc.GaussianBlur(grayMat, grayMat, new Size(5, 5), 0);

                // Detect faces
                faceDetector.detectMultiScale(grayMat, faces);

                for (Rect face : faces.toArray()) {
                    Imgproc.rectangle(frameMat, face.tl(), face.br(), new Scalar(0, 255, 0), 2);
                }

                // Convert to BufferedImage and display
                BufferedImage img = matToBufferedImage(frameMat);
                if (img != null) {
                    imageLabel.setIcon(new ImageIcon(img));
                    imageLabel.repaint();
                }

                try {
                    Thread.sleep(33);
                } catch (InterruptedException e) {
                    break;
                }
            }

            capture.release();
        }).start();
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_3BYTE_BGR;
        if (mat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        }

        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b);

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), b);

        return image;
    }
}
