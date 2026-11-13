#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    // 1. 读图
    Mat src = imread("image.png");
    if (src.empty())
    {
        cerr << "无法读取图片 image.png" << endl;
        return -1;
    }

    // 2. 裁掉图像四周的黑边框(约8像素)
    Rect roi(8, 8, src.cols - 16, src.rows - 16);
    src = src(roi).clone();

    // 3. 灰度化 + 高斯滤波去噪
    Mat gray, blurred;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 4. 阈值分割: 回形针为深色, 取反后为白色前景
    //    使用 OTSU 自动确定阈值
    Mat binary;
    threshold(blurred, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 5. 形态学闭运算: 填补回形针内部线圈间的缝隙, 使每个回形针成为一个连通域
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat closed;
    morphologyEx(binary, closed, MORPH_CLOSE, kernel, Point(-1, -1), 2);

    // 6. 连通域分析
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(closed, labels, stats, centroids, 8, CV_32S);

    // 7. 按面积过滤噪声, 统计回形针数量并标注
    Mat result = src.clone();
    int count = 0;
    for (int i = 1; i < numLabels; i++)   // 0 为背景
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 2000)                   // 过滤小噪声
            continue;

        count++;

        Rect box(stats.at<int>(i, CC_STAT_LEFT),
                 stats.at<int>(i, CC_STAT_TOP),
                 stats.at<int>(i, CC_STAT_WIDTH),
                 stats.at<int>(i, CC_STAT_HEIGHT));
        rectangle(result, box, Scalar(0, 0, 255), 2);
        putText(result, to_string(count),
                Point(box.x + box.width / 2 - 10, box.y + box.height / 2),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);

        cout << "目标 " << count << ": 面积=" << area
             << " 位置=(" << box.x << "," << box.y << ")" << endl;
    }

    // 7. 显示结果
    cout << "\n========== 回形针总数: " << count << " ==========" << endl;

    putText(result, "Count: " + to_string(count), Point(20, 40),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 3);

    imshow("原图", src);
    imshow("二值化", binary);
    imshow("闭运算", closed);
    imshow("计数结果", result);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
