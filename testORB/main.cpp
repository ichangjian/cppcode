#include "opencv2/opencv.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include "undistorter.h"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
int surf();
int orb();
Mat Undistort(Mat imageD, Mat K, Mat D);
void initUndistortMap(cv::Mat &map1, cv::Mat &map2, double fScale, double Kalibr[8], int imageWidth, int imageHeight);
void test();
void liftProjective(Point2f src, Point2f &dst, Mat K, Mat D);
void clb();

int main(int argc, char **argv)
{
    // test();
    clb();
    // surf();
    // orb();
    return 0;
}
int estimatWithRansacBase(
    const std::vector<cv::KeyPoint> &keyPoint1,
    const std::vector<cv::KeyPoint> &keyPoint2,
    const std::vector<cv::DMatch> &matchePoints,
    std::vector<cv::DMatch> &goodMatchePoints,
    cv::Mat_<double> &H,
    double inlier_thresh = 25);

void test()
{
    Mat image_object = imread("129371025720.png", IMREAD_GRAYSCALE);
    Mat image_scene = imread("129704247699.png", IMREAD_GRAYSCALE);
    //    k2:
    //    mu: 290.03672785595984
    //    mv: 289.70361075706387
    //    u0: 323.1621621450474
    //    v0: 197.5696586534049
    Eigen::Vector2d focalLength(290.03672785595984, 289.70361075706387);
    Eigen::Vector2d principalPoint(323.1621621450474, 197.5696586534049);
    Eigen::Vector2i resolution(image_scene.cols, image_scene.rows);
    Eigen::Vector4d distCoeffs_RadTan(-0.01847757657533866, 0.0575172937703169, -0.06496298696673658, 0.02593645307703224);
    undistorter::PinholeGeometry camera(focalLength, principalPoint, resolution, undistorter::EquidistantDistortion::create(distCoeffs_RadTan));

    ////////////////////////////
    // create the undistorter
    ////////////////////////////
    double alpha = 1.0, //alphe=0.0: all pixels valid, alpha=1.0: no pixels lost
        scale = 2.0;
    int interpolation = cv::INTER_LINEAR;
    undistorter::PinholeUndistorter undistorter(camera, alpha, scale, interpolation);
    Mat undist_image;
    undistorter.undistortImage(image_object, undist_image);
    imwrite("undist_image.png",undist_image);
  cv::imshow("WINDOW_NAME", undist_image);
  cv::waitKey();
}

void clb()
{
    ifstream inFile;
    string path = "/media/hi/warehouse/slamdata-test/clb/slamdata-clb-1513/cam0/";
    inFile.open((path+"../loop.txt").c_str());
    string fileName;
    while (inFile >> fileName)
    {
        // Mat view = imread(path+fileName+".png");
        Mat view = imread("20190604142457.jpg");
        Mat viewGray;
        cvtColor(view, viewGray, CV_BGR2GRAY);
        vector<Point2f> pointbuf;
        Size boardSize(6, 4);
        bool found = findChessboardCorners(view, boardSize, pointbuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        if (found)
        {
            cornerSubPix(viewGray, pointbuf, Size(7, 7), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
            for (size_t i = 0; i < pointbuf.size(); i++)
            {
                char s[10];
                sprintf(s, "%d", i);
                putText(view, s, pointbuf[i], 1, 1, Scalar(0, 0, 255));
            }
        }
        namedWindow("clb", WINDOW_NORMAL);
        imshow("clb", view);
        waitKey(0);
    }
}

int surf()
{
    Mat image_object = imread("2823552185837.png", IMREAD_GRAYSCALE);
    Mat image_scene = imread("2823585508025.png", IMREAD_GRAYSCALE);

    //判断图像是否加载成功
    if (image_object.empty() || image_scene.empty())
    {
        cout << "图像加载失败";
        return -1;
    }
    else
        cout << "图像加载成功..." << endl
             << endl;
    //    k2: -0.01847757657533866
    //    k3: 0.0575172937703169
    //    k4: -0.06496298696673658
    //    k5: 0.02593645307703224
    //    mu: 290.03672785595984
    //    mv: 289.70361075706387
    //    u0: 323.1621621450474
    //    v0: 197.5696586534049
    Mat camera = (Mat_<double>(3, 3) << 290.03672785595984, 0, 323.1621621450474, 0, 289.70361075706387, 197.5696586534049, 0, 0, 1);
    Mat dist = (Mat_<double>(1, 4) << -0.01847757657533866, 0.0575172937703169, -0.06496298696673658, 0.02593645307703224);

    //   Camera model: pinhole
    //   Focal length: [291.19607705698485, 290.93041771031943]
    //   Principal point: [322.3420020588464, 195.87797636693793]
    //   Distortion model: equidistant
    //   Distortion coefficients: [0.006695702265625288, -0.03159047755071519, 0.05796030866063177, -0.0170372498385688]
    // Mat camera=(Mat_<double>(3,3)<<291.19607705698485,0,322.3420020588464,0,290.93041771031943,195.87797636693793,0,0,1);
    // Mat dist=(Mat_<double>(1,4)<<0.006695702265625288, -0.03159047755071519, 0.05796030866063177, -0.0170372498385688);
    Mat tp, ftp_obj,ftp_scen;
    ftp_obj = Undistort(image_object, camera, dist);
    ftp_scen = Undistort(image_scene, camera, dist);
    // imshow("a", image_object);
    // undistort(image_object, tp, camera, dist);
    // image_object = tp.clone();

    // imshow("b", image_object);
    // imshow("c", ftp);
    // waitKey();
    // undistort(image_scene, tp, camera, dist);
    // image_scene = tp.clone();
    circle(image_object, Point(image_object.cols / 2, image_object.rows / 2), 50, Scalar(0), -1);
    circle(image_scene, Point(image_object.cols / 2, image_object.rows / 2), 50, Scalar(0), -1);
    //检测特征点
    const int minHessian = 700;

    Ptr<SIFT> detector = SIFT::create();
    Ptr<GFTTDetector> gftt = GFTTDetector::create(200, 0.1, 30);
    vector<KeyPoint> keypoints_object, keypoints_scene;
    detector->detect(image_object, keypoints_object);
    detector->detect(image_scene, keypoints_scene);
    // vector<Point2f>corner;
    // goodFeaturesToTrack(image_object,corner,500,0.1,30);
    // KeyPoint kp=KeyPoint();
    // for (size_t i = 0; i < corner.size(); i++)
    // {
    //     kp.pt=corner[i];
    //    keypoints_object.push_back(kp);
    // }
    clock_t time1 = clock();
    //计算特征点描述子
    Ptr<SIFT> extractor = SIFT::create();
    Mat descriptors_object, descriptors_scene;
    extractor->compute(image_object, keypoints_object, descriptors_object);
    extractor->compute(image_scene, keypoints_scene, descriptors_scene);

    //使用FLANN进行特征点匹配
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);

    //计算匹配点之间最大和最小距离
    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < descriptors_object.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
        {
            min_dist = dist;
        }
        else if (dist > max_dist)
        {
            max_dist = dist;
        }
    }
    printf("Max dist: %f \n", max_dist);
    printf("Min dist: %f \n", min_dist);

    //绘制“好”的匹配点
    vector<DMatch> bgood_matches, good_matches;
    Mat_<double> H;
    cout << "aaaaaa" << estimatWithRansacBase(keypoints_object, keypoints_scene, matches, good_matches, H);

    clock_t time2 = clock();
    cout << "\ttime=" << double(time2 - time1) / CLOCKS_PER_SEC << endl;
    // for (int i = 0; i < descriptors_object.rows; i++)
    // {
    //     if (matches[i].distance < 3 * min_dist)
    //     {
    //         good_matches.push_back(matches[i]);
    //     }
    // }
    Mat image_matches;
    drawMatches(image_object, keypoints_object, image_scene, keypoints_scene, good_matches, image_matches,
                Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //定位“好”的匹配点
    vector<Point2f> obj;
    vector<Point2f> scene;
    Mat merg(image_object.rows*2,image_object.cols*2,CV_8UC1,Scalar(0));
    merg(Rect(0,0,image_object.cols,image_object.rows))+=ftp_scen;
    merg(Rect(image_object.cols,image_object.rows,image_object.cols,image_object.rows))+=ftp_obj;
    for (int i = 0; i < good_matches.size(); i++)
    {
        //DMathch类型中queryIdx是指match中第一个数组的索引,keyPoint类型中pt指的是当前点坐标
        Point2f dst1,dst2;
        liftProjective(keypoints_scene[good_matches[i].trainIdx].pt,dst1,camera, dist);
        scene.push_back(dst1);
        liftProjective(keypoints_object[good_matches[i].queryIdx].pt,dst2,camera, dist);
        obj.push_back(dst2);
        Rect rect(0,0,image_object.cols,image_object.rows);
        // Mat merg1=merg.clone();
        // if(rect.contains(dst1)&&rect.contains(dst2))
        // {
        //     line(merg1, dst1, dst2 + Point2f(image_object.cols, image_object.rows), Scalar(255));
        // imshow("merg",merg1);
        // // waitKey();

        // }
        // obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        // scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }
    vector<uchar> status;
    Mat F = cv::findFundamentalMat(scene, obj, cv::FM_RANSAC, 3.0, 0.999, status);

    H = findHomography(scene, obj, CV_RANSAC,1, status,1000, 0.999);

        vector<cv::Mat> r, t, n;
        cv::decomposeHomographyMat(H,camera,r,t,n);
        for (size_t i = 0; i < r.size(); i++)
        {cout<<r[i]<<"\t"<<t[i]<<endl;
        }
        
    vector<Point2f> obj_corners(4), scene_corners(4), oobj;
    obj_corners[0] = cvPoint(0, 0);
    obj_corners[1] = cvPoint(image_object.cols, 0);
    obj_corners[2] = cvPoint(image_object.cols, image_object.rows);
    obj_corners[3] = cvPoint(0, image_object.rows);

    perspectiveTransform(obj_corners, scene_corners, H);
    perspectiveTransform(scene, oobj, H);
    Point2f sum(0,0);
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i] == 1)
        {
            sum+=Point2f(abs(obj[i].x-oobj[i].x),abs(obj[i].y-oobj[i].y));
            cout << obj[i] << "\t" << oobj[i] << endl;
        }
    }
cout<<"============"<<sum.x/status.size()<<"\t"<<sum.y/status.size()<<"\t"<<status.size()<<endl;
    // //绘制角点之间的直线
    // line(image_matches, scene_corners[0] + Point2f(image_object.cols, 0),
    //      scene_corners[1] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
    // line(image_matches, scene_corners[1] + Point2f(image_object.cols, 0),
    //      scene_corners[2] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
    // line(image_matches, scene_corners[2] + Point2f(image_object.cols, 0),
    //      scene_corners[3] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
    // line(image_matches, scene_corners[3] + Point2f(image_object.cols, 0),
    //      scene_corners[0] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
    //绘制角点之间的直线
    line(image_matches, scene_corners[0],
         scene_corners[1], Scalar(0, 0, 255), 2);
    line(image_matches, scene_corners[1],
         scene_corners[2], Scalar(0, 0, 255), 2);
    line(image_matches, scene_corners[2],
         scene_corners[3], Scalar(0, 0, 255), 2);
    line(image_matches, scene_corners[3],
         scene_corners[0], Scalar(0, 0, 255), 2);
    Mat wp;
    warpPerspective(ftp_scen, wp, H, image_object.size());
    imwrite("scen.png",wp);
    // warpPerspective(ftp_obj, wp, H, image_object.size());
    imwrite("obj.png",ftp_obj);

    //输出图像
    namedWindow("匹配图像", WINDOW_NORMAL);
    imshow("匹配图像", image_matches);
    namedWindow("匹配图像1", WINDOW_NORMAL);
    imshow("匹配图像1", (wp + ftp_obj) / 2);
    waitKey(0);
}

int orb()
{
    //读取要匹配的两张图像
    Mat img_1 = imread("c.png", 1);
    Mat img_2 = imread("a.jpg", 1);

    //初始化
    //首先创建两个关键点数组，用于存放两张图像的关键点，数组元素是KeyPoint类型
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    //创建两张图像的描述子，类型是Mat类型
    Mat descriptors_1, descriptors_2;

    //创建一个ORB类型指针orb，ORB类是继承自Feature2D类
    //class CV_EXPORTS_W ORB : public Feature2D
    //这里看一下create()源码：参数较多，不介绍。
    //creat()方法所有参数都有默认值，返回static　Ptr<ORB>类型。
    /*
    CV_WRAP static Ptr<ORB> create(int nfeatures=500,
                                   float scaleFactor=1.2f,
                                   int nlevels=8,
                                   int edgeThreshold=31,
                                   int firstLevel=0,
                                   int WTA_K=2,
                                   int scoreType=ORB::HARRIS_SCORE,
                                   int patchSize=31,
                                   int fastThreshold=20);
    */
    //所以这里的语句就是创建一个Ptr<ORB>类型的orb，用于接收ORB类中create()函数的返回值
    Ptr<ORB> orb = ORB::create(2000);

    //第一步：检测Oriented FAST角点位置.
    //detect是Feature2D中的方法，orb是子类指针，可以调用
    //看一下detect()方法的原型参数：需要检测的图像，关键点数组，第三个参数为默认值
    /*
    CV_WRAP virtual void detect( InputArray image,
                                 CV_OUT std::vector<KeyPoint>& keypoints,
                                 InputArray mask=noArray() );
    */
    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);

    //第二步：根据角点位置计算BRIEF描述子
    //compute是Feature2D中的方法，orb是子类指针，可以调用
    //看一下compute()原型参数：图像，图像的关键点数组，Mat类型的描述子
    /*
    CV_WRAP virtual void compute( InputArray image,
                                  CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors );
    */
    orb->compute(img_1, keypoints_1, descriptors_1);
    orb->compute(img_2, keypoints_2, descriptors_2);

    //定义输出检测特征点的图片。
    Mat outimg1;
    //drawKeypoints()函数原型参数：原图，原图关键点，带有关键点的输出图像，后面两个为默认值
    /*
    CV_EXPORTS_W void drawKeypoints( InputArray image,
                                     const std::vector<KeyPoint>& keypoints,
                                     InputOutputArray outImage,
                                     const Scalar& color=Scalar::all(-1),
                                     int flags=DrawMatchesFlags::DEFAULT );
    */
    //注意看，这里并没有用到描述子，描述子的作用是用于后面的关键点筛选。
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    namedWindow("ORB特征点", WINDOW_NORMAL);
    imshow("ORB特征点", outimg1);

    //第三步：对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离

    //创建一个匹配点数组，用于承接匹配出的DMatch，其实叫match_points_array更为贴切。matches类型为数组，元素类型为DMatch
    vector<DMatch> matches;

    //创建一个BFMatcher匹配器，BFMatcher类构造函数如下：两个参数都有默认值，但是第一个距离类型下面使用的并不是默认值，而是汉明距离
    //CV_WRAP BFMatcher( int normType=NORM_L2, bool crossCheck=false );
    BFMatcher matcher(NORM_HAMMING);

    //调用matcher的match方法进行匹配,这里用到了描述子，没有用关键点。
    //匹配出来的结果写入上方定义的matches[]数组中
    matcher.match(descriptors_1, descriptors_2, matches);

    //第四步：遍历matches[]数组，找出匹配点的最大距离和最小距离，用于后面的匹配点筛选。
    //这里的距离是上方求出的汉明距离数组，汉明距离表征了两个匹配的相似程度，所以也就找出了最相似和最不相似的两组点之间的距离。
    double min_dist = 0, max_dist = 0; //定义距离

    for (int i = 0; i < descriptors_1.rows; ++i) //遍历
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("Max dist: %f\n", max_dist);
    printf("Min dist: %f\n", min_dist);

    //第五步：根据最小距离，对匹配点进行筛选，
    //当描述自之间的距离大于两倍的min_dist，即认为匹配有误，舍弃掉。
    //但是有时最小距离非常小，比如趋近于0了，所以这样就会导致min_dist到2*min_dist之间没有几个匹配。
    // 所以，在2*min_dist小于30的时候，就取30当上限值，小于30即可，不用2*min_dist这个值了
    std::vector<DMatch> good_matches;
    for (int j = 0; j < descriptors_1.rows; ++j)
    {
        if (matches[j].distance <= max(2 * min_dist, 30.0))
            good_matches.push_back(matches[j]);
    }

    //第六步：绘制匹配结果

    Mat img_match; //所有匹配点图
    //这里看一下drawMatches()原型参数，简单用法就是：图1，图1关键点，图2，图2关键点，匹配数组，承接图像，后面的有默认值
    /*
    CV_EXPORTS_W void drawMatches( InputArray img1,
                                   const std::vector<KeyPoint>& keypoints1,
                                   InputArray img2,
                                   const std::vector<KeyPoint>& keypoints2,
                                   const std::vector<DMatch>& matches1to2,
                                   InputOutputArray outImg,
                                   const Scalar& matchColor=Scalar::all(-1),
                                   const Scalar& singlePointColor=Scalar::all(-1),
                                   const std::vector<char>& matchesMask=std::vector<char>(),
                                   int flags=DrawMatchesFlags::DEFAULT );
    */

    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    namedWindow("所有匹配点对", WINDOW_NORMAL);
    imshow("所有匹配点对", img_match);

    Mat img_goodmatch; //筛选后的匹配点图
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    namedWindow("筛选后的匹配点对", WINDOW_NORMAL);
    imshow("筛选后的匹配点对", img_goodmatch);

    waitKey(0);
}

int ComputeAffine(
    const std::vector<cv::Point_<double> > &pt1,
    const std::vector<cv::Point_<double> > &pt2,
    cv::Mat_<double> &H)
{
    if (pt1.size() != pt2.size() || pt1.size() < 4)
    {
        return -1;
    }

    cv::Mat_<double> A = cv::Mat_<double>(pt1.size() * 2, 6);
    cv::Mat_<double> B = cv::Mat_<double>(pt1.size() * 2, 1);
    cv::Mat_<double> C = cv::Mat_<double>(6, 1);
    double *p1 = A.ptr<double>(0);
    double *p2 = B.ptr<double>(0);

    for (int i = 0; i < pt1.size(); i++)
    {
        *(p1++) = pt2[i].x;
        *(p1++) = pt2[i].y;
        *(p1++) = 1.0F;
        *(p1++) = 0.0F;
        *(p1++) = 0.0F;
        *(p1++) = 0.0F;
        *(p1++) = 0.0F;
        *(p1++) = 0.0F;
        *(p1++) = 0.0F;
        *(p1++) = pt2[i].x;
        *(p1++) = pt2[i].y;
        *(p1++) = 1.0F;
        *(p2++) = pt1[i].x;
        *(p2++) = pt1[i].y;
    }

    cv::Mat At = A.t();
    C = (At * A).inv() * At * B;

    H = cv::Mat_<double>(3, 3);
    p1 = C.ptr<double>(0);
    p2 = H.ptr<double>(0);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            *(p2++) = *(p1++);
        }
    }

    H[2][0] = 0.0F;
    H[2][1] = 0.0F;
    H[2][2] = 1.0F;

    return 0;
}

int estimatWithRansacBase(
    const std::vector<cv::KeyPoint> &keyPoint1,
    const std::vector<cv::KeyPoint> &keyPoint2,
    const std::vector<cv::DMatch> &matchePoints,
    std::vector<cv::DMatch> &goodMatchePoints,
    cv::Mat_<double> &H,
    double inlier_thresh)
{
    std::vector<int> tempIndex;
    size_t tempNum;
    int maxLoopCount = 1500;
    int loopCount = 0;
    cv::Mat_<double> htemp;
    std::vector<cv::Point_<double> > pt1(4);
    std::vector<cv::Point_<double> > pt2(4);

    size_t corspNum = matchePoints.size();

    if (corspNum <= 4)
        return -1;

    tempIndex.resize((corspNum / 100 + 1) * 100);
    goodMatchePoints.clear();

    while (loopCount < maxLoopCount)
    {
        int sample[4];
        for (int i = 0; i < 4; i++)
        {
            static int s = 0;
            int j;

            if (s++ == 0)
                srand(0);

            int pos = (int)((double)corspNum * rand() / (RAND_MAX + 1.0F));
            for (j = 0; j < i; j++)
            {
                if (pos == sample[j])
                    break;
            }
            if (j < i)
            {
                i--;
                continue;
            }
            else
            {
                sample[i] = pos;
            }
            pt1[i].x = keyPoint1[matchePoints[pos].queryIdx].pt.x;
            pt1[i].y = keyPoint1[matchePoints[pos].queryIdx].pt.y;
            pt2[i].x = keyPoint2[matchePoints[pos].trainIdx].pt.x;
            pt2[i].y = keyPoint2[matchePoints[pos].trainIdx].pt.y;
        }

        //if ((!IsGoodSample(pt1)) || (!IsGoodSample(pt2))) { loopCount++; continue; }

        if (ComputeAffine(pt1, pt2, htemp) < 0)
        {
            loopCount++;
            continue;
        }
        double *htempptr = htemp.ptr<double>(0);
        tempNum = 0;
        for (size_t i = 0; i < corspNum; i++)
        {
            if (i == 18)
                i = i;
            double xx2, yy2, ww2, err;
            xx2 = htempptr[0] * keyPoint2[matchePoints[i].trainIdx].pt.x + htempptr[1] * keyPoint2[matchePoints[i].trainIdx].pt.y + htempptr[2];
            yy2 = htempptr[3] * keyPoint2[matchePoints[i].trainIdx].pt.x + htempptr[4] * keyPoint2[matchePoints[i].trainIdx].pt.y + htempptr[5];
            ww2 = htempptr[6] * keyPoint2[matchePoints[i].trainIdx].pt.x + htempptr[7] * keyPoint2[matchePoints[i].trainIdx].pt.y + htempptr[8];
            xx2 /= ww2;
            yy2 /= ww2;
            err = (xx2 - keyPoint1[matchePoints[i].queryIdx].pt.x) * (xx2 - keyPoint1[matchePoints[i].queryIdx].pt.x) +
                  (yy2 - keyPoint1[matchePoints[i].queryIdx].pt.y) * (yy2 - keyPoint1[matchePoints[i].queryIdx].pt.y);
            if (err < inlier_thresh)
            {
                tempIndex[tempNum] = i;
                tempNum++;
            }
            else
            {
                err = err;
            }
        }

        if (tempNum > goodMatchePoints.size())
        {
            H = htemp.clone();

            goodMatchePoints.resize(tempNum);
            for (size_t i = 0; i < tempNum; i++)
            {
                goodMatchePoints[i] = matchePoints[tempIndex[i]];
            }

            static const double p = log(1.0F - 0.99F);
            double e = static_cast<double>(tempNum) / static_cast<double>(corspNum);
            double e_thresh = pow((1 - pow(10, -2.0F / maxLoopCount)), 0.25);
            if (e > e_thresh)
            {
                double N = (p / log(1 - e * e * e * e));
                maxLoopCount = static_cast<int>(N);
            }
        }
        loopCount++;
    }

    pt1.resize(goodMatchePoints.size());
    pt2.resize(goodMatchePoints.size());
    for (int i = 0; i < goodMatchePoints.size(); i++)
    {
        pt1[i].x = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
        pt1[i].y = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
        pt2[i].x = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
        pt2[i].y = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
    }
    if (ComputeAffine(pt1, pt2, H) < 0)
    {
        return -2;
    }

    return 0;
}

void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P, double Kalibr[8])
{
    double mu = Kalibr[0], mv = Kalibr[1], u0 = Kalibr[2], v0 = Kalibr[3], k2 = Kalibr[4], k3 = Kalibr[5], k4 = Kalibr[6], k5 = Kalibr[7];

    double m_inv_K11, m_inv_K22, m_inv_K13, m_inv_K23;
    m_inv_K11 = 1.0 / mu;
    m_inv_K13 = -u0 / mu;
    m_inv_K22 = 1.0 / mv;
    m_inv_K23 = -v0 / mv;
    // Lift points to normalised plane
    Eigen::Vector2d p_u;
    p_u << m_inv_K11 * p(0) + m_inv_K13,
        m_inv_K22 * p(1) + m_inv_K23;

    // Obtain a projective ray
    double theta, phi;

    int interation_num = 30;

    double p_u_norm = p_u.norm();
    if (p_u_norm < 1e-10)
    {
        phi = 0.0;
    }
    else
    {
        phi = atan2(p_u(1), p_u(0));
    }
    theta = p_u_norm;

    //iteration to get final theta
    double theta_d = p_u_norm;
    if (theta_d > 1e-8)
    {
        double theta_old = theta_d;
        for (int j = 0; j < interation_num; j++)
        {
            //double  theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
            //theta = theta_d / (1 + mParameters.k2() * theta2 + mParameters.k3() * theta4 + mParameters.k4() * theta6 + mParameters.k5() * theta8);

            double theta2 = theta * theta, theta3 = theta2 * theta, theta5 = theta3 * theta2, theta7 = theta5 * theta2, theta9 = theta7 * theta2;
            theta = theta_d - (k2 * theta3 + k3 * theta5 + k4 * theta7 + k5 * theta9);

            if (std::fabs(theta_old - theta) < 1e-10)
                break;

            theta_old = theta;
            //std::cout<<theta<<" ";
        }
        //std::cout<<"\n";
    }

    P(0) = sin(theta) * cos(phi);
    P(1) = sin(theta) * sin(phi);
    P(2) = cos(theta);
}
void liftProjective(Point2f src, Point2f &dst, Mat K, Mat D)
{
    double Kalibr[8];
    Kalibr[0] = K.at<double>(0, 0);
    Kalibr[1] = K.at<double>(1, 1);
    Kalibr[2] = K.at<double>(0, 2);
    Kalibr[3] = K.at<double>(1, 2);
    Kalibr[4] = D.at<double>(0);
    Kalibr[5] = D.at<double>(1);
    Kalibr[6] = D.at<double>(2);
    Kalibr[7] = D.at<double>(3);
    Eigen::Vector2d a(src.x, src.y);
    Eigen::Vector3d b;
    liftProjective(a, b, Kalibr);
    dst.x = b.x() / b.z() * Kalibr[0] + Kalibr[2];
    dst.y = b.y() / b.z() * Kalibr[1] + Kalibr[3];
}
Mat Undistort(Mat imageD, Mat K, Mat D)
{
    int ROW = imageD.rows;
    int COL = imageD.cols;
    int FOCAL_LENGTH = 290;
    double Kalibr[8];
    Kalibr[0] = K.at<double>(0, 0);
    Kalibr[1] = K.at<double>(1, 1);
    Kalibr[2] = K.at<double>(0, 2);
    Kalibr[3] = K.at<double>(1, 2);
    Kalibr[4] = D.at<double>(0);
    Kalibr[5] = D.at<double>(1);
    Kalibr[6] = D.at<double>(2);
    Kalibr[7] = D.at<double>(3);
    cv::Mat undistortedImg; //(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    Mat map1(ROW, COL, CV_32FC1), map2(ROW, COL, CV_32FC1);
    // initUndistortMap(map1, map2, 1, Kalibr, COL, ROW);
    // remap(imageD, undistortedImg, map1, map2, INTER_LINEAR | WARP_INVERSE_MAP);

    // cv::imshow("name", undistortedImg);
    // imwrite("aa.png", undistortedImg);
    // return undistortedImg;

    vector<Eigen::Vector2d> distortedp, undistortedp;
        for (int i = 0; i < COL; i++)
    {
        for (int j = 0; j < ROW; j++)
        {
            Point2f dst;
            liftProjective(Point2f(i,j), dst, K,D);
            int x =dst.x;
            int y = dst.y;
            if (x > 0 && y > 0 && x < COL && y < ROW)
            {
                map1.at<float>(y, x) = i;
                map2.at<float>(y, x) = j;
            }
        }
    }
    // for (int i = 0; i < COL; i++)
    // {
    //     for (int j = 0; j < ROW; j++)
    //     {
    //         Eigen::Vector2d a(i, j);
    //         Eigen::Vector3d b;
    //         liftProjective(a, b, Kalibr);
    //         distortedp.push_back(a);
    //         undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
    //         map1.at<float>(j, i) = b.x() / b.z() * FOCAL_LENGTH + COL / 2;
    //         map2.at<float>(j, i) = b.y() / b.z() * FOCAL_LENGTH + ROW / 2;
    //         //    int x = b.x() / b.z() * FOCAL_LENGTH + COL / 2;
    //         //     int y = b.y() / b.z() * FOCAL_LENGTH + ROW / 2;
    //         //     if(x>0&&y>0&&x<COL&&y<ROW)
    //         //     {
    //         //     map1.at<float>(y, x) = i;
    //         //     map2.at<float>(y, x) = j;

    //         //     }
    //     }
    // }
    // convertMaps(map1,map2,map1,map2,CV_32FC1);
    // for (int i = 0; i < int(undistortedp.size()); i++)
    // {
    //     cv::Mat pp(3, 1, CV_32FC1);
    //     pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
    //     pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
    //     pp.at<float>(2, 0) = 1.0;
    //     //cout << trackerData[0].K << endl;
    //     //LOGD("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    //     //LOGD("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    //     if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
    //     {
    //         undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = imageD.at<uchar>(distortedp[i].y(), distortedp[i].x());
    //     }
    // }
    remap(imageD, undistortedImg, map1, map2, INTER_LINEAR );

    cv::imshow("name", undistortedImg);
    imwrite("aa.png", undistortedImg);
    return undistortedImg;
}

void backprojectSymmetric(const Eigen::Vector2d &p_u, double &theta, double &phi, double Kalibr[8])
{
    double tol = 1e-10;
    double p_u_norm = p_u.norm();
    double mu = Kalibr[0], mv = Kalibr[1], u0 = Kalibr[2], v0 = Kalibr[3], k2 = Kalibr[4], k3 = Kalibr[5], k4 = Kalibr[6], k5 = Kalibr[7];

    if (p_u_norm < 1e-10)
    {
        phi = 0.0;
    }
    else
    {
        phi = atan2(p_u(1), p_u(0));
    }

    int npow = 9;
    if (k5 == 0.0)
    {
        npow -= 2;
    }
    if (k4 == 0.0)
    {
        npow -= 2;
    }
    if (k3 == 0.0)
    {
        npow -= 2;
    }
    if (k2 == 0.0)
    {
        npow -= 2;
    }

    Eigen::MatrixXd coeffs(npow + 1, 1);
    coeffs.setZero();
    coeffs(0) = -p_u_norm;
    coeffs(1) = 1.0;

    if (npow >= 3)
    {
        coeffs(3) = k2;
    }
    if (npow >= 5)
    {
        coeffs(5) = k3;
    }
    if (npow >= 7)
    {
        coeffs(7) = k4;
    }
    if (npow >= 9)
    {
        coeffs(9) = k5;
    }

    if (npow == 1)
    {
        theta = p_u_norm;
    }
    else
    {
        // Get eigenvalues of companion matrix corresponding to polynomial.
        // Eigenvalues correspond to roots of polynomial.
        Eigen::MatrixXd A(npow, npow);
        A.setZero();
        A.block(1, 0, npow - 1, npow - 1).setIdentity();
        A.col(npow - 1) = -coeffs.block(0, 0, npow, 1) / coeffs(npow);

        Eigen::EigenSolver<Eigen::MatrixXd> es(A);
        Eigen::MatrixXcd eigval = es.eigenvalues();

        std::vector<double> thetas;
        for (int i = 0; i < eigval.rows(); ++i)
        {
            if (fabs(eigval(i).imag()) > tol)
            {
                continue;
            }

            double t = eigval(i).real();

            if (t < -tol)
            {
                continue;
            }
            else if (t < 0.0)
            {
                t = 0.0;
            }

            thetas.push_back(t);
        }

        if (thetas.empty())
        {
            theta = p_u_norm;
        }
        else
        {
            theta = *std::min_element(thetas.begin(), thetas.end());
        }
    }
}
template<typename T> T r(T k2, T k3, T k4, T k5, T theta)
{
    // k1 = 1
    return theta +
           k2 * theta * theta * theta +
           k3 * theta * theta * theta * theta * theta +
           k4 * theta * theta * theta * theta * theta * theta * theta +
           k5 * theta * theta * theta * theta * theta * theta * theta * theta * theta;
}
void initUndistortMap(cv::Mat &map1, cv::Mat &map2, double fScale, double Kalibr[8], int imageWidth, int imageHeight)
{
    cv::Size imageSize(imageWidth, imageHeight);
    double mu = Kalibr[0], mv = Kalibr[1], u0 = Kalibr[2], v0 = Kalibr[3], k2 = Kalibr[4], k3 = Kalibr[5], k4 = Kalibr[6], k5 = Kalibr[7];

    double m_inv_K11, m_inv_K22, m_inv_K13, m_inv_K23;
    m_inv_K11 = 1.0 / mu;
    m_inv_K13 = -u0 / mu;
    m_inv_K22 = 1.0 / mv;
    m_inv_K23 = -v0 / mv;
    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            double theta, phi;
            backprojectSymmetric(Eigen::Vector2d(mx_u, my_u), theta, phi, Kalibr);

            Eigen::Vector3d P;
            P << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);

            Eigen::Vector2d p;
            // spaceToPlane(P, p);
            {
                // spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const

                double theta = acos(P(2) / P.norm());
                double phi = atan2(P(1), P(0));

                Eigen::Vector2d p_u = r(k2, k3, k4, k5, theta) * Eigen::Vector2d(cos(phi), sin(phi));

                // Apply generalised projection matrix
                p << mu * p_u(0) + u0,
                    mv * p_u(1) + v0;
            }

            mapX.at<float>(v, u) = p(0);
            mapY.at<float>(v, u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}