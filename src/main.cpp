#include <iostream>
#include <memory>
#include<vector>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <algorithm>

#include "detector.h"
#include "utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "Eigen/Dense"

using namespace cv;
using namespace std;

void listDir(const char *name, vector<string> &fileNames, bool lastSlash)
{
    DIR *dir;
    struct dirent *entry;
    struct stat statbuf;
    struct tm      *tm;
    time_t rawtime;
    if (!(dir = opendir(name)))
    {
        cout<<"Couldn't open the file or dir"<<name<<"\n";
        return;
    }
    if (!(entry = readdir(dir)))
    {
        cout<<"Couldn't read the file or dir"<<name<<"\n";
        return;
    }

    do
    {
        string slash="";
        if(!lastSlash)
            slash = "/";

        string parent(name);
        string file(entry->d_name);
        string final = parent + slash + file;
        if(stat(final.c_str(), &statbuf)==-1)
        {
            cout<<"Couldn't get the stat info of file or dir: "<<final<<"\n";
            return;
        }
        if (entry->d_type == DT_DIR) //its a directory
        {
            //skip the . and .. directory
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            listDir(final.c_str(), fileNames, false);
        }
        else // it is a file
        {
            fileNames.push_back(final);
        }
    }while (entry = readdir(dir));
    closedir(dir);
}




std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


void Demo(cv::Mat& img,
        const std::vector<std::vector<Detection>>& detections,
        const std::vector<std::string>& class_names,
        bool label = true) {

    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                        cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                        cv::Point(box.tl().x + s_size.width, box.tl().y),
                        cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
            }
        }
    }

    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", img);
    cv::waitKey(300);
}

int max_value = 255;
int threshold_value = 100;


int main(int argc, const char* argv[]) {
    // set device type - CPU/GPU
    torch::DeviceType device_type;
    device_type = torch::kCPU;

    // load class names from dataset for visualization
    std::vector<std::string> class_names = LoadNames("../weights/classes.names");
    if (class_names.empty()) {
        return -1;
    }

    // load network
    std::string weights = "../weights/best_cpu.torchscript";
    auto detector = Detector(weights, device_type);

    // load input color image
    string directoryPath = "../images/";
    vector<string> fileNames;
    listDir(directoryPath.c_str(), fileNames, true);

    // load input depth image
    string directoryPath_depth = "../depth/";
    vector<string> fileNames_depth;
    listDir(directoryPath_depth.c_str(), fileNames_depth, true);

    // run once to warm up
    //std::cout << "Run once on empty image" << std::endl;
    //auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    //detector.Run(temp_img, 1.0f, 1.0f);

    // set up threshold
    float conf_thres = 0.5;
    float iou_thres = 0.4;

    //钥匙孔的三维坐标
    float keyhole_z = 0;
    float keyhole_x = 0;
    float keyhole_y = 0;

    //钥匙孔到把手的方向向量
    float handle_ray_z = 0;
    float handle_ray_x = 0;
    float handle_ray_y = 0;

    //把手上转轴的的三维坐标
    float handle_z = 0;
    float handle_x = 0;
    float handle_y = 0;

    //把手下夹持部位的的三维坐标
    float handle_to_grip_x = 0;
    float handle_to_grip_y = 0;
    float handle_to_grip_z = 0;

    //钥匙孔坐标系
    Eigen::Matrix3d keyhole_co;

    //把手上侧坐标系
    Eigen::Matrix3d handle_top;

    //把手下侧坐标系
    Eigen::Matrix3d handle_bottom;

    for (int i = 1; i < fileNames.size(); ++i) {
        //read one color image
        cv::Mat img = cv::imread(fileNames[i]);
        if (img.empty()) {
            std::cerr << "Error loading the image!\n";
            return -1;
        }

        // inference
        auto result = detector.Run(img, conf_thres, iou_thres);
        cv::Mat image_handle = img(Range(result[0][0].bbox.y, result[0][0].bbox.y + result[0][0].bbox.height),
                                   Range(result[0][0].bbox.x, result[0][0].bbox.x + result[0][0].bbox.width));
        //key hole detction
        //转为灰度图
        cv::Mat grayImage;
        cvtColor(image_handle, grayImage, COLOR_BGR2GRAY);

        //转换为二值图
        cv::Mat binaryImage;
        threshold(grayImage, binaryImage, 110, 255, THRESH_BINARY);//old thresh 110

        vector<vector<Point> > contours;  //这里面将来会存储找到的边界的（x,y）坐标
        findContours(binaryImage,
                     contours,    //轮廓的数组
                     RETR_EXTERNAL,   //获取外轮廓
                     CHAIN_APPROX_NONE);  //获取每个轮廓的每个像素

        //cv::Mat result_keyhole(binaryImage.size(), CV_8U, Scalar(255));

        //移除过长或过短的轮廓
        int cmin = 100; //最小轮廓长度  //相机和把手之间离的远也没有必要去识别定位。
        int cmax = 1000;    //最大轮廓
        // vector < vector<Point> >::const_iterator itc = contours.begin();
        vector<vector<Point> >::iterator itc = contours.begin();

        while (itc != contours.end()) {
            if (itc->size() < cmin || itc->size() > cmax)
                itc = contours.erase(itc);
                // itc = contours.erase(itc);
            else
                ++itc;
        }

        //提取连通区域的轮廓，根据外形轮廓确定最小长方形
        // 遍历轮廓显示旋转矩形框
        std::vector<std::vector<cv::Point>> tmpContours;    // 创建一个InputArrayOfArrays 类型的点集
        std::vector<cv::RotatedRect> rotaterect_vector;

        for (int i = 0; i < contours.size(); ++i) {
            cv::RotatedRect rotatedrect = cv::minAreaRect(cv::Mat(contours[i]));
            // 存储旋转矩形的四个点
            // 提取旋转矩形的四个角点
            cv::Point2f ps[4];  //Point2f(x,y)中的x代表在图像中的列，y代表图像中的行。
            rotatedrect.points(ps);


            if (rotatedrect.size.height / rotatedrect.size.width < 3 &&
                rotatedrect.size.width / rotatedrect.size.height < 3) {
                std::vector<cv::Point> contour;
                for (int i = 0; i != 4; ++i) {
                    contour.emplace_back(cv::Point2i(ps[i]));
                }
                // 插入到轮廓容器中
                tmpContours.insert(tmpContours.end(), contour);
                rotaterect_vector.push_back(rotatedrect);

            } else {
                continue;
            }

        }
        // tmpContours里面是56个矩形的顶点坐标（x,y）。
        drawContours(image_handle, tmpContours, -1, Scalar(0, 0, 255), 1, 16);  // 填充mask

        //read one depth image
        cv::Mat img_d = cv::imread(fileNames_depth[i], 2);
        if (img_d.empty()) {
            std::cerr << "Error loading the depth image!\n";
            return -1;
        }

        vector<Image_Point> image_points;///创建一个存放像素点三维坐标的容器

        if (rotaterect_vector.size() == 1) {
            //coordinate translation
            for (int i = 0; i < 4; ++i) {
                tmpContours[0][i].x = tmpContours[0][i].x + result[0][0].bbox.x;
                tmpContours[0][i].y = tmpContours[0][i].y + result[0][0].bbox.y;
            }
            ///根据有深度值的点估计平面
            //获取有深度的点并转换成三维坐标
            int handle_center_x = result[0][0].bbox.x + result[0][0].bbox.width / 2;//门把手中心坐标x
            int handle_center_y = result[0][0].bbox.y + result[0][0].bbox.height / 2;//门把手中心坐标y

            float x_average = 0;
            float y_average = 0;
            float z_average = 0;
            for (int x = handle_center_x - result[0][0].bbox.width / 4;
                 x < handle_center_x + result[0][0].bbox.width / 4; ++x) {
                for (int y = handle_center_y - result[0][0].bbox.height / 4;
                     y < handle_center_y + result[0][0].bbox.height / 4; ++y) {

                    //if(rot_x < rotaterect_vector[0].size.width && rot_y < rotaterect_vector[0].size.height){
                    if (img_d.at<ushort>(y, x) != 0) {
                        Image_Point imagePoint_in;
                        imagePoint_in.z = float(img_d.at<ushort>(y, x)) / 1000;
                        imagePoint_in.x = ((x - 655.5707397460938) / 908.211181640625) * imagePoint_in.z;
                        imagePoint_in.y = ((y - 359.12640380859375) / 908.211181640625) * imagePoint_in.z;

                        z_average = z_average + imagePoint_in.z;
                        x_average = x_average + imagePoint_in.x;
                        y_average = y_average + imagePoint_in.y;

                        image_points.push_back(imagePoint_in);
                        //d_in = d_in + img_d.at<ushort>(y,x);
                    }
                    //}
                }
            }
            ///计算特征值、特征向量，最小特征值对应的特征向量就是平面法向量，选取平面上一点，即可表示平面
            if (image_points.size() > 3) {
                z_average = z_average / image_points.size();
                x_average = x_average / image_points.size();
                y_average = y_average / image_points.size();

                Eigen::Vector3d center(x_average, y_average, z_average);

                Eigen::Vector3d tmp(0, 0, 0);

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                for (int i = 0; i < image_points.size(); ++i) {
                    tmp(0) = image_points[i].x;
                    tmp(1) = image_points[i].y;
                    tmp(2) = image_points[i].z;
                    Eigen::Matrix<double, 3, 1> tmpzeromean = tmp - center;

                    covMat = covMat + tmpzeromean * tmpzeromean.transpose();

                }
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> sase(covMat);


                Eigen::Vector3d direction = sase.eigenvectors().col(0);

                ///根据中心像素坐标计算与平面的交点即中心三维坐标
                //get the keyhole center
                int center_x = (tmpContours[0][0].x + tmpContours[0][2].x) / 2;
                int center_y = (tmpContours[0][0].y + tmpContours[0][2].y) / 2;

                Eigen::Vector3d ray((center_x - 655.5707397460938) / 908.211181640625,
                                    (center_y - 359.12640380859375) / 908.211181640625, 1);

                keyhole_x = direction.dot(center) /
                           (direction(0) + ray(1) / ray(0) * direction(1) + ray(2) / ray(0) * direction(2));
                keyhole_y = ray(1) / ray(0) * keyhole_x;
                keyhole_z = ray(2) / ray(0) * keyhole_x;

                cout << keyhole_x << " "
                     << keyhole_y << " "
                     << keyhole_z << endl;

                //lsd detection
                Mat dst;
                Canny(image_handle, dst, 50, 200, 3); // Apply canny edge
                // Create and LSD detector with standard or no refinement.
                Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_ADV);

                double start = double(getTickCount());
                vector<Vec4f> lines_std;
                // Detect the lines
                ls->detect(dst, lines_std);

                vector<Vec4f> lines_std_filtered;
                Vec4f line_in(0.0, 0.0, 0.0, 0.0);
                Vec2f line_result(0, 0);
                int j = 0;
                cout << "numbers: " << lines_std.size() << endl;
                for (int i = 0; i < lines_std.size(); ++i) {
                    float len = sqrt((lines_std[i][0] - lines_std[i][2]) * (lines_std[i][0] - lines_std[i][2]) +
                                     (lines_std[i][1] - lines_std[i][3]) * (lines_std[i][1] - lines_std[i][3]));
                    if (len < 90)
                        continue;
                    else {
                        line_in[0] = lines_std[i][0];
                        line_in[1] = lines_std[i][1];
                        line_in[2] = lines_std[i][2];
                        line_in[3] = lines_std[i][3];
                        lines_std_filtered.push_back(line_in);

                        if (i != lines_std.size() - 1) {
                            line_result[0] = line_result[0] - abs(line_in[0] - line_in[2]);
                            line_result[1] = line_result[1] - abs(line_in[1] - line_in[3]);
                        } else {
                            line_result[0] = line_result[0] / (j + 1);
                            line_result[1] = line_result[1] / (j + 1);
                        }
                        j++;
                    }
                }


                double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                std::cout << "It took " << duration_ms << " ms." << std::endl;

                // Show found lines
                Mat drawnLines(image_handle);
                ls->drawSegments(drawnLines, lines_std_filtered);
                imshow("Standard refinement", drawnLines);


                // visualize detections
                int key_hole_center_x = (tmpContours[0][0].x + tmpContours[0][2].x) / 2 ;
                int key_hole_center_y = (tmpContours[0][0].y + tmpContours[0][2].y) / 2 ;


                int handle_center_x_cal = key_hole_center_x + 300 * line_result[0] / sqrt(line_result[0] * line_result[0] +
                                                                                          line_result[1] * line_result[1]);
                int handle_center_y_cal = key_hole_center_y + 300 * line_result[1] / sqrt(line_result[0] * line_result[0] +
                                                                                          line_result[1] * line_result[1]);

                cv::line(img, Point(handle_center_x_cal, handle_center_y_cal), Point(key_hole_center_x, key_hole_center_y),
                         Scalar(0, 0, 255), 1, 16, 0);

                Eigen::Vector3d ray_handle((handle_center_x_cal - 655.5707397460938) / 908.211181640625,
                                           (handle_center_y_cal - 359.12640380859375) / 908.211181640625, 1);

                handle_ray_x = direction.dot(center) /
                            (direction(0) + ray_handle(1) / ray_handle(0) * direction(1) + ray_handle(2) / ray_handle(0) * direction(2));
                handle_ray_y = ray_handle(1) / ray_handle(0) * handle_ray_x;
                handle_ray_z = ray_handle(2) / ray_handle(0) * handle_ray_x;


                cout << handle_ray_x << " "
                     << handle_ray_y << " "
                     << handle_ray_z << endl;

                Eigen::Vector3d keyhole2handle(handle_ray_x-keyhole_x,handle_ray_y-keyhole_y,handle_ray_z-keyhole_z);

                handle_x = keyhole_x + 0.0972 * keyhole2handle(0)/ sqrt(keyhole2handle(0)*keyhole2handle(0) + keyhole2handle(1)*keyhole2handle(1) + keyhole2handle(2)*keyhole2handle(2));
                handle_y = keyhole_y + 0.0972 * keyhole2handle(1)/ sqrt(keyhole2handle(0)*keyhole2handle(0) + keyhole2handle(1)*keyhole2handle(1) + keyhole2handle(2)*keyhole2handle(2));;
                handle_z = keyhole_z + 0.0972 * keyhole2handle(2)/ sqrt(keyhole2handle(0)*keyhole2handle(0) + keyhole2handle(1)*keyhole2handle(1) + keyhole2handle(2)*keyhole2handle(2));;

                cout << handle_x << " "
                     << handle_y << " "
                     << handle_z << endl;


                handle_to_grip_x = keyhole_x + 0.0472 * keyhole2handle(0)/ sqrt(keyhole2handle(0)*keyhole2handle(0) + keyhole2handle(1)*keyhole2handle(1) + keyhole2handle(2)*keyhole2handle(2));
                handle_to_grip_y = keyhole_y + 0.0472 * keyhole2handle(1)/ sqrt(keyhole2handle(0)*keyhole2handle(0) + keyhole2handle(1)*keyhole2handle(1) + keyhole2handle(2)*keyhole2handle(2));;
                handle_to_grip_z = keyhole_z + 0.0472 * keyhole2handle(2)/ sqrt(keyhole2handle(0)*keyhole2handle(0) + keyhole2handle(1)*keyhole2handle(1) + keyhole2handle(2)*keyhole2handle(2));;

                cout << handle_to_grip_x << " "
                     << handle_to_grip_y << " "
                     << handle_to_grip_z << endl;

                keyhole2handle.normalized();
                direction.normalize();
                Eigen::Vector3d cross = keyhole2handle.cross(direction);

                keyhole_co.row(0) = cross;
                keyhole_co.row(1) = keyhole2handle;
                keyhole_co.row(2)= direction;

                Eigen::Quaterniond quaterniond(keyhole_co);
                quaterniond.normalize();
                cout<<quaterniond.x()<<" "
                    <<quaterniond.y()<<" "
                    <<quaterniond.z()<<" "
                    <<quaterniond.w()<<endl;
            }
        }

        Demo(img, result, class_names);
    }
        cv::destroyAllWindows();
        return 0;
}

