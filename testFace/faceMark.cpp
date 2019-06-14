#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::face;
void dlib_face();
void lbf_face();
int main()
{
    dlib_face();
    // lbf_face();
    return 0;
}

void dlib_face()
{
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        int argc=3;
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return ;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

        image_window win, win_faces;
        // Loop over all the images provided on the command line.
        for (int i = 2; i < argc; ++i)
        {
            cout << "processing image " << "argv[i]" << endl;
            array2d<rgb_pixel> img;
            load_image(img, "2.png");
            // Make the image larger so we can detect small faces.
            // pyramid_up(img);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            std::vector<dlib::rectangle> dets = detector(img);
            cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);
                cout << "number of parts: " << shape.num_parts() << endl;
                cout << "pixel position of first part:  " << shape.part(0) << endl;
                cout << "pixel position of second part: " << shape.part(1) << endl;
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                shapes.push_back(shape);
            }

            // Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections(shapes));
            Mat image=imread("2.png");
            for (size_t j = 0; j < shapes.size(); j++)
            {
            
            for (size_t i = 0; i < shapes[j].num_parts(); i++)
            {
                circle(image,Point2f(shapes[j].part(i).x(),shapes[j].part(i).y()),3,Scalar(0,0,255),1);
            }
            }
            
            imwrite("tp1.png",image);
            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            dlib::array<array2d<rgb_pixel>> face_chips;
            extract_image_chips(img, get_face_chip_details(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));

            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        }
    }
    catch (exception &e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}
void lbf_face()
{
    // [1]Haar Face Detector
    //CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
    // [2]LBP Face Detector
    CascadeClassifier faceDetector("lbpcascade_frontalface_improved.xml");

    // 创建Facemark类的对象
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // 加载人脸检测器模型
    facemark->loadModel("lbfmodel.yaml");

    // 设置网络摄像头用来捕获视频

    // 存储视频帧和灰度图的变量
    Mat frame, gray;
    frame = imread("scen.png");
    // frame=frame(Rect(0,0,1500,1000));
    // resize(frame,frame,Size(),0.5,0.5);
    // 读取帧

    {

        // 存储人脸矩形框的容器
        std::vector<Rect> faces;
        // 将视频帧转换至灰度图, 因为Face Detector的输入是灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 人脸检测
        faceDetector.detectMultiScale(gray, faces);
        // facemark->getFaces(gray, faces);
        // 人脸关键点的容器
        std::vector<std::vector<Point2f> > landmarks;

        // 运行人脸关键点检测器（landmark detector）
        bool success = facemark->fit(frame, faces, landmarks);

        if (success)
        {
            // 如果成功, 在视频帧上绘制关键点
            for (int i = 0; i < landmarks.size(); i++)
            {
                // 自定义绘制人脸特征点函数, 可绘制人脸特征点形状/轮廓
                // drawLandmarks(frame, landmarks[i]);
                // OpenCV自带绘制人脸关键点函数: drawFacemarks
                drawFacemarks(frame, landmarks[i], Scalar(0, 0, 255));
            }
        }

        // 显示结果
        namedWindow("Facial Landmark Detection", WINDOW_NORMAL);
        imshow("Facial Landmark Detection", frame);
        imwrite("face2.jpg", frame);
        waitKey();
    }
    return;
}