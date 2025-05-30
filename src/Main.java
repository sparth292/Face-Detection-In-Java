import javax.swing.*; // GUI window banvayla javax.swing vaparla
import java.awt.image.BufferedImage; // BufferedImage ha Java image la represent karayla
import org.opencv.core.*; // OpenCV madhil basic classes import kele
import org.opencv.videoio.VideoCapture; // Video capture sathi he import kele
import org.opencv.imgproc.Imgproc; // Image processing sathi functions
import org.opencv.objdetect.CascadeClassifier; // Face detection sathi classifier vaparto
import org.opencv.core.MatOfRect; // Rectangles che group face detection sathi

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // OpenCV chi native library load kartoy
    }

    public static void main(String[] args) {
        System.out.println("OpenCV loaded successfully. Version: " + Core.VERSION); // Console var OpenCV load zala ka te dakhavtoy

        JFrame frame = new JFrame("Face Detection"); // GUI window create keli
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Window close zalyavar program band kara
        JLabel imageLabel = new JLabel(); // Image display karayla label create kelay
        frame.add(imageLabel); // Frame madhe label add kelay
        frame.setSize(640, 480); // Window chi size set keli
        frame.setVisible(true); // Window visible keli

        // Face detection sathi cascade classifier load kartoy
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        if (faceDetector.empty()) { // Classifier load zala ka nahi check kartoy
            System.out.println(" Failed to load cascade classifier!"); // Load nahi zala tar error
            return;
        } else {
            System.out.println(" Cascade loaded successfully."); // Success message
        }

        VideoCapture capture = new VideoCapture(0); // Webcam access karayla 0 (default camera) open kartoy
        if (!capture.isOpened()) { // Camera open zala ka nahi check kartoy
            System.out.println(" Cannot open webcam."); // Error dakhavtoy
            return;
        }

        // Thread madhe live camera frame gheun face detect kartoy
        new Thread(() -> {
            Mat frameMat = new Mat(); // Original frame store karayla mat object
            Mat grayMat = new Mat(); // Gray image sathi mat object
            MatOfRect faces = new MatOfRect(); // Saglya detect zalele faces store karayla

            while (true) {
                if (!capture.read(frameMat)) continue; // Camera kadun image read hot nahi tar continue

                // Color image la gray madhe convert karto
                Imgproc.cvtColor(frameMat, grayMat, Imgproc.COLOR_BGR2GRAY);

                // Gray image chi contrast vadhalay sathi histogram equalization
                Imgproc.equalizeHist(grayMat, grayMat);

                // Thodi noise kami karayla Gaussian blur vaparlo
                Imgproc.GaussianBlur(grayMat, grayMat, new Size(5, 5), 0);

                // Face detection call karto
                faceDetector.detectMultiScale(grayMat, faces);

                // Saglya faces var green rectangle draw karto
                for (Rect face : faces.toArray()) {
                    Imgproc.rectangle(frameMat, face.tl(), face.br(), new Scalar(0, 255, 0), 2);
                }

                // Frame image convert karto BufferedImage madhe
                BufferedImage img = matToBufferedImage(frameMat);
                if (img != null) {
                    imageLabel.setIcon(new ImageIcon(img)); // Label var image set karto
                    imageLabel.repaint(); // GUI refresh karto
                }

                try {
                    Thread.sleep(33); // Approx 30 FPS sathi thread la delay detoy
                } catch (InterruptedException e) {
                    break;
                }
            }

            capture.release(); // Camera release karto
        }).start();
    }

    // OpenCV Mat object la BufferedImage madhe convert karaycha method
    private static BufferedImage matToBufferedImage(Mat mat) {
        // Color fix karayla BGR to RGB convert karto
        Mat rgbMat = new Mat();
        Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGR2RGB);

        int type = BufferedImage.TYPE_3BYTE_BGR;
        if (rgbMat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY; // Ekach channel asel tar gray image
        }

        // Data la array madhe copy karto
        int bufferSize = rgbMat.channels() * rgbMat.cols() * rgbMat.rows();
        byte[] b = new byte[bufferSize];
        rgbMat.get(0, 0, b);

        // BufferedImage create karun data set karto
        BufferedImage image = new BufferedImage(rgbMat.cols(), rgbMat.rows(), type);
        image.getRaster().setDataElements(0, 0, rgbMat.cols(), rgbMat.rows(), b);

        return image; // BufferedImage return karto
    }
}
