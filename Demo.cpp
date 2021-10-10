#include "templMatching.h"

int main() {
    //�ȶ�ȡģ��
	cv::Mat src =
		cv::imread("E:/SmartMore/WorkSpace/LineModAcc/linemodacc/linemodacc/temp/temp1.png", 0);    
	float scale = 0.5;
    resize(src, src, cv::Size(src.cols * scale, src.rows * scale), 0, 0, cv::INTER_AREA);
    cv::Mat maskTest = cv::Mat(src.size(), CV_8UC1, {255});

    //�������������
    int num_features = 200;
    float weakThreshold = 100;
    float strongThreshold = 200;
    std::vector<int> T{2, 4};
    std::vector<int> vRotate{0, 360, 1};
    Detector detector(num_features, T, weakThreshold, strongThreshold, vRotate);

    //��ȡ������
    detector.addTemplate(src, maskTest);

    //����������
    std::string dir = "./";
    detector.save(dir);

    //����������
    detector.load(dir);

    //��ȡ����ͼ��
    cv::Mat matchSrc =
        cv::imread("E:/SmartMore/WorkSpace/LineModAcc/linemodacc/linemodacc/match/match16.png", 0);
    resize(matchSrc, matchSrc, cv::Size(matchSrc.cols * scale, matchSrc.rows * scale), 0, 0, cv::INTER_AREA);

    //ִ��ƥ��
    detector.matchTemplate(matchSrc, 1);

    return 0;
}
