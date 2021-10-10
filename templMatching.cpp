#include "templMatching.h"

//扩散量化
static inline int getLabel(int quantized) {
    switch (quantized) {
        case 0:
            return 0;
        case 1:
            return 1;
        case 2:
            return 2;
        case 4:
            return 3;
        case 8:
            return 4;
        case 16:
            return 5;
        case 32:
            return 6;
        case 64:
            return 7;
        case 128:
            return 8;
        default:
            CV_Error(cv::Error::StsBadArg, "Invalid value of quantized parameter");
            return -1;  // avoid warning
    }
}

//筛选特征点
bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                                   std::vector<Feature>& features, size_t num_features,
                                                   float distance) {
	if (distance == 0) {
		return false;
	}
    features.clear();
    float distance_sq = distance * distance;
    size_t i = 0;

    bool first_select = true;

    while (true) {
        Candidate c = candidates[i];

        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j) {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep) features.push_back(c.f);

        if (++i == (int)candidates.size()) {
            bool num_ok = features.size() >= num_features;

            if (first_select) {
                if (num_ok) {
                    features.clear();
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                } else {
                    first_select = false;
                }
            }

            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
            if (num_ok || distance < 2) {
                break;
            }
        }
    }
    return true;
}

//梯度量化
void hysteresisGradient(cv::Mat& magnitude, cv::Mat& quantized_angle, cv::Mat& angle, float threshold) {
    cv::Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    for (int r = 0; r < quantized_unfiltered.rows; ++r) {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    for (int r = 1; r < angle.rows - 1; ++r) {
        uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c) {
            quant_r[c] &= 7;
        }
    }

    quantized_angle = cv::Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r) {
        float* mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c) {
            if (mag_r[c] > threshold) {
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar* patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                int max_votes = 0;
                int index = -1;
                for (size_t i = 0; i < 8; ++i) {
                    if (max_votes < histogram[i]) {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                static const int NEIGHBOR_THRESHOLD = 6;
                if (max_votes >= NEIGHBOR_THRESHOLD) quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

//梯度计算
static void quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude, cv::Mat& angle, cv::Mat& angle_ori,
                                  float threshold) {
    cv::Mat smoothed = src;

    static const int KERNEL_SIZE = 3;

    if (src.channels() == 1) {
        cv::Mat sobel_dx, sobel_dy, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;

    } else {
        magnitude.create(src.size(), CV_32F);

        cv::Size size = src.size();
        cv::Mat sobel_3dx;
        cv::Mat sobel_3dy;
        cv::Mat sobel_dx(size, CV_32F);
        cv::Mat sobel_dy(size, CV_32F);
        cv::Mat sobel_ag;

        Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

        short* ptrx = (short*)sobel_3dx.data;
        short* ptry = (short*)sobel_3dy.data;
        float* ptr0x = (float*)sobel_dx.data;
        float* ptr0y = (float*)sobel_dy.data;
        float* ptrmg = (float*)magnitude.data;

        const int length1 = static_cast<const int>(sobel_3dx.step1());
        const int length2 = static_cast<const int>(sobel_3dy.step1());
        const int length3 = static_cast<const int>(sobel_dx.step1());
        const int length4 = static_cast<const int>(sobel_dy.step1());
        const int length5 = static_cast<const int>(magnitude.step1());
        const int length0 = sobel_3dy.cols * 3;

        for (int r = 0; r < sobel_3dy.rows; ++r) {
            int ind = 0;

            for (size_t i = 0; i < length0; i += 3) {
                int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
                int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
                int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

                if (mag1 >= mag2 && mag1 >= mag3) {
                    ptr0x[ind] = ptrx[i];
                    ptr0y[ind] = ptry[i];
                    ptrmg[ind] = (float)mag1;
                } else if (mag2 >= mag1 && mag2 >= mag3) {
                    ptr0x[ind] = ptrx[i + 1];
                    ptr0y[ind] = ptry[i + 1];
                    ptrmg[ind] = (float)mag2;
                } else {
                    ptr0x[ind] = ptrx[i + 2];
                    ptr0y[ind] = ptry[i + 2];
                    ptrmg[ind] = (float)mag3;
                }
                ++ind;
            }
            ptrx += length1;
            ptry += length2;
            ptr0x += length3;
            ptr0y += length4;
            ptrmg += length5;
        }

        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;
    }
}

void ColorGradientPyramid::update() {
    if (!mask.empty()) {
        cv::Mat next_mask;
        resize(mask, next_mask, src.size(), 0.0, 0.0, cv::INTER_NEAREST);
        mask = next_mask;
    }
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
}

void ColorGradientPyramid::pyrDown(int index) {
    num_features /= index;
    ++pyramid_level;

    int pyrNum = index / 2;
    cv::Mat next_src;
    cv::Size size;
    for (int i = 0; i < pyrNum; i++) {
        size = cv::Size(src.cols / 2, src.rows / 2);
        cv::pyrDown(src, next_src, size);
        src = next_src;
    }

    if (!mask.empty()) {
        cv::Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, cv::INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::quantize(cv::Mat& dst) const {
    dst = cv::Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template& templ, cv::Mat draw) const {
    cv::Mat t;
    draw.copyTo(t);
    cv::Mat local_mask;
    if (!mask.empty()) {
        erode(mask, local_mask, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    }

    std::vector<Candidate> candidates;
	std::vector<std::vector<Candidate>> vCandidate(8);
	std::vector<std::vector<Feature>> vFeature(8);

    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 5;

    cv::Mat k = cv::Mat(angle.size(), CV_8UC1, cv::Scalar(255));

    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0 + nms_kernel_size / 2; r < magnitude.rows - nms_kernel_size / 2; ++r) {
        const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0 + nms_kernel_size / 2; c < magnitude.cols - nms_kernel_size / 2; ++c) {
            if (no_mask || mask_r[c]) {
                float score = 0;
                if (magnitude_valid.at<uchar>(r, c) > 0 && magnitude.at<float>(r, c) > threshold_sq) {
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++) {
                        for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) {
                            if (r_offset == 0 && c_offset == 0) continue;

                            if (score < magnitude.at<float>(r + r_offset, c + c_offset)) {
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                        if (!is_max) break;
                    }

                    if (is_max) {
                        for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++) {
                            for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) {
                                if (r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r + r_offset, c + c_offset) = 0;
                            }
                        }
                    }

                    if (score > threshold_sq && angle.at<uchar>(r, c) > 0) {
                        for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++) {
                            for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) {
                                if (r_offset == 0 && c_offset == 0) continue;
                                k.at<uchar>(r + r_offset, c + c_offset) = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle.at<uchar>(r, c) > 0) {
					int nowLabel = getLabel(angle.at<uchar>(r, c));
					if (nowLabel != 0) {
						vCandidate[nowLabel - 1].emplace_back(Candidate(c, r, nowLabel, score));
						vCandidate[nowLabel - 1].back().f.theta = angle_ori.at<float>(r, c);
					}
                    //candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                   // candidates.back().f.theta = angle_ori.at<float>(r, c);
                }
            }
        }
    }
	for (int cnt = 0; cnt < 8; cnt++) {
		if (vCandidate[cnt].size() <= 2) {
			std::cout << "too few features, abort" << std::endl;
		}
		std::stable_sort(vCandidate[cnt].begin(), vCandidate[cnt].end());

		float distance = static_cast<float>(vCandidate[cnt].size() / (num_features/8));

		if (!selectScatteredFeatures(vCandidate[cnt], vFeature[cnt], num_features/8, distance)) {
			//return false;
		}
		for (auto k : vFeature[cnt]) {
			templ.features.emplace_back(k);
		}
		
	}

    /*if (candidates.size() < num_features) {
        if (candidates.size() <= 4) {
            std::cout << "too few features, abort" << std::endl;
            return false;
        }
        std::cout << "have no enough features, exaustive mode" << std::endl;
    }

    std::stable_sort(candidates.begin(), candidates.end());

    float distance = static_cast<float>(candidates.size() / num_features);

    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance)) {
        return false;
    }
*/
    for (auto k : templ.features) {
        cv::circle(t, cv::Point(k.x, k.y), 2, cv::Scalar(0, 255, 255), -1, 8, 0);
    }

    return true;
}

Detector::Detector() {
    this->modality = cv::makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(2);
    T_at_level.push_back(4);
}

Detector::Detector(std::vector<int> T) {
    this->modality = cv::makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, float weak_thresh, float strong_threash, std::vector<int> vRotate) {
    this->modality = cv::makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash,
                   std::vector<int> vRotate) {
    this->modality = cv::makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
    this->vRotate = vRotate;
}

cv::Mat Detector::transform(cv::Mat src, float angle, float scale = 1) {
    cv::Mat dst;
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(src, dst, rot_mat, src.size());
    return dst;
}

bool Detector::addTemplate(const cv::Mat& source, const cv::Mat& object_mask) {
    srcImage = source;
    int num_features = modality->num_features;
    int rotateNum = (vRotate[1] - vRotate[0]) / vRotate[2];

    pyramid_levels = T_at_level.size();
    cv::Mat dst, dstMask;
    source.copyTo(dst);
    object_mask.copyTo(dstMask);
    cv::Ptr<ColorGradientPyramid> qp = modality->process(dst, dstMask);

    TemplatePyramid tp;
    tp.resize(rotateNum);

    if (num_features > 0) qp->num_features = num_features;

    for (int l = 0; l < pyramid_levels; l++) {
        for (int i = 0; i < rotateNum; i++) {
            qp->num_features = num_features;
            qp->src = source;
            qp->mask = object_mask;
            qp->src = transform(qp->src, i * vRotate[2], 1.0);
            qp->mask = transform(qp->mask, i * vRotate[2], 1.0);

            qp->pyrDown(T_at_level[l]);

            if (i == 0) {
                sumVal.push_back(sum(qp->src)[0]);
            }
			tp[i].features.clear();
            bool success = qp->extractTemplate(tp[i], qp->src);
            if (!success) return false;
        }

        class_templates.emplace_back(tp);
    }

    return true;
}

bool Detector::matchTemplate(const cv::Mat& source, int nameNumb)
{
	clock_t startTime, endTime;
	startTime = clock();
	float offset_x = 0;
	float offset_y = 0;
	float localAngle = 0;
	float similarity = 0;
	for (int loopVTPNum = class_templates.size() - 1; loopVTPNum >= 0; loopVTPNum--) {
		similarity = 0;
		cv::Mat matchSrc;
		source.copyTo(matchSrc);

		int n = class_templates[loopVTPNum].size();
		if (loopVTPNum != class_templates.size() - 1) {
			localAngle = localAngle >= 5 ? localAngle - 5 : n - (5 - localAngle);
			n = localAngle + 10;
			/*if (n >= class_templates[loopVTPNum].size()) {
				n = n - class_templates[loopVTPNum].size();
			}*/
		}
		//对待测图像进行金字塔降采样
		cv::Size window = srcSize;
		for (int j = 0; j < T_at_level[loopVTPNum]; j += 2) {
			cv::pyrDown(matchSrc, matchSrc);
			window.width /= 2;
			window.height /= 2;
		}
		int pairNum = 0;
		//计算全图的梯度方向
		cv::Mat gradientVal, gradientAngle, gradientAngle_roi;
		quantizedOrientations(matchSrc, gradientVal, gradientAngle, gradientAngle_roi, 70);

		if (loopVTPNum == class_templates.size() - 1) {

			//只在底层进行操作
			//对待测图像的全图方向进行分类
			std::vector<std::vector<cv::Point2f>> matchSrcAnglePoint(8);
			for (int i = 0; i < gradientAngle.cols - 1; i++) {
				for (int j = 0; j < gradientAngle.rows - 1; j++) {
					int labelVal = getLabel(gradientAngle.at<uchar>(j, i));
					//std::cout << i << "," << j << std::endl;
					if (labelVal > 0 && gradientVal.at<float>(j, i) > 50000) {
						matchSrcAnglePoint[labelVal - 1].emplace_back(cv::Point2f(i, j));
					}
				}
			}
			//按照坐标进行排序
			for (int k = 0; k < matchSrcAnglePoint.size(); k++) {
				sort(matchSrcAnglePoint[k].begin(), matchSrcAnglePoint[k].end(), [](cv::Point2f& a, cv::Point2f& b) {
					if (a.x < b.x) {
						return true;
					}
					else if (a.x == b.x) {
						return a.y < b.y;
					}
					return false;
				});
			}
			float x_t=0, y_t=0, angle_t=0;

			//角度匹配

			for (int loopRotateNum = 0; loopRotateNum < n; loopRotateNum++) {
				//对当前角度的特征点进行方向分类
				std::vector<std::vector<cv::Point2f>> templAnglePoint(8);
				std::vector<int> maxSize(8);
				for (auto k : class_templates[loopVTPNum][loopRotateNum].features) {
					if (k.label > 0) {
						templAnglePoint[k.label - 1].emplace_back(cv::Point2f(k.x, k.y));
					}
					maxSize[k.label - 1]++;
				}
				//找到元素最多的一个方向
				int maxPosition1 = max_element(maxSize.begin(), maxSize.end()) - maxSize.begin();
				//同样排序
				for (int k = 0; k < templAnglePoint.size(); k++) {
					sort(templAnglePoint[k].begin(), templAnglePoint[k].end(), [](cv::Point2f& a, cv::Point2f& b) {
						if (a.x < b.x) {
							return true;
						}
						else if (a.x == b.x) {
							return a.y < b.y;
						}
						return false;
					});
				}

				//设置阈值
				int gradientThreshold;
				for (int maxPosition = 0; maxPosition < 8; maxPosition++) {
					if (templAnglePoint[maxPosition].size() == 0) {
						continue;
					}
					float offsetX;
					float offsetY;
					//设置边界值
					int index = 500;
					//设置kernel
					int kernel = 5;
					//设置匹配的点数
					int matchPairNum = 0;
					bool flag = false;
					//先找到第一个匹配点
					for (int loopMatchSrcPoint = 0; loopMatchSrcPoint < matchSrcAnglePoint[maxPosition].size() && (!flag); loopMatchSrcPoint++) {
						offsetX = matchSrcAnglePoint[maxPosition][loopMatchSrcPoint].x - templAnglePoint[maxPosition][0].x;
						offsetY = matchSrcAnglePoint[maxPosition][loopMatchSrcPoint].y - templAnglePoint[maxPosition][0].y;
						flag = true;

						//判断该offset是否在前4个内有效
						for (int count = 1; count < templAnglePoint[maxPosition].size() - 1 && count < index; count++) {
							flag = isExitPoint(cv::Point2f(templAnglePoint[maxPosition][count].x + offsetX,
								templAnglePoint[maxPosition][count].y + offsetY),
								matchSrcAnglePoint[maxPosition], kernel);
							if (!flag) {
								break;
							}
						}
					}

					//利用offset进行匹配
					for (int loopRoi = 0; loopRoi < templAnglePoint.size() && flag; loopRoi++) {
						for (auto k : templAnglePoint[loopRoi]) {
							if (isExitPoint(cv::Point2f(k.x + offsetX, k.y + offsetY), matchSrcAnglePoint[loopRoi], kernel)) {
								matchPairNum++;
							}
						}
					}
					if (similarity < (float)matchPairNum / (float)class_templates[loopVTPNum][loopRotateNum].features.size()) {
						similarity = (float)matchPairNum / (float)class_templates[loopVTPNum][loopRotateNum].features.size();
						x_t = offsetX;
						y_t = offsetY;
						angle_t = loopRotateNum;
					}
				}
				//std::cout << loopRotateNum << std::endl;
			}
			offset_x = x_t;
			offset_y = y_t;
			localAngle = angle_t;
			for (auto k : class_templates[loopVTPNum][localAngle].features) {
				cv::circle(matchSrc, cv::Point(k.x + offset_x, k.y + offset_y), 3, cv::Scalar(255, 255, 255), -1, 8, 0);
			}
		}
		else {
			int i_temp = 0;
			int j_temp = 0;
			int tAngle = 0;
			offset_x = (offset_x * 2 - 10) > 0 ? (offset_x * 2 - 10) : (offset_x * 2);
			offset_y = (offset_y * 2 - 10) > 0 ? (offset_y * 2 - 10) : (offset_y * 2);
			//统计该模板下的方向总和
			for (int i = offset_x; i < matchSrc.cols - window.width && i < offset_x + 20; i++) {
				for (int j = offset_y; j < matchSrc.rows - window.height && j < offset_y + 20; j++) {
					int pairNumPre = 0;
					// st

					cv::Mat matchSrcMagnitudeROI = gradientVal(cv::Rect(i, j, window.width, window.height));
					cv::Mat matchSrcROI = matchSrc(cv::Rect(i, j, window.width, window.height));
					// ed

					cv::Mat matchSrcAngleROI = gradientAngle(cv::Rect(i, j, window.width, window.height));
					if (cv::sum(matchSrcROI)[0] < sumVal[loopVTPNum] * 0.75) {
						//continue;
					}
					int directionIndex = 1;
					for (int loopRotateNum = localAngle; loopRotateNum < n; loopRotateNum++) {
						pairNum = 0;

						cv::Mat tt; //绘图
						matchSrc(cv::Rect(i, j, window.width, window.height)).copyTo(tt);
						for (auto k : class_templates[loopVTPNum][loopRotateNum % class_templates[loopVTPNum].size()].features) {
							cv::circle(tt, cv::Point(k.x, k.y), 2, cv::Scalar(0), -1, 8, 0);
						}
						bool flag = true;
						int kernel = 2;
						for (auto k : class_templates[loopVTPNum][loopRotateNum % class_templates[loopVTPNum].size()].features) {
							flag = false;
							float agVal = matchSrcMagnitudeROI.at<float>(k.y, k.x);
							for (int c = 0; c < kernel; c++) {
								for (int r = 0; r < kernel; r++) {
									if (k.y + r >= matchSrcMagnitudeROI.rows || k.x + c >= matchSrcMagnitudeROI.cols) {
										continue;
									}
									float agValMax = matchSrcMagnitudeROI.at<float>(k.y + r, k.x + c);
									int oriVal = matchSrcAngleROI.at<uchar>(k.y + r, k.x + c);
									int diffOri = abs(getLabel(oriVal) - k.label);
									if (abs(agValMax - agVal) < 50000 && diffOri < 2 && agVal > 50000) {
										flag = true;
									}
								}
							}
							if (flag) {
								cv::circle(tt, cv::Point(k.x, k.y), 2, cv::Scalar(255, 255, 255), -1, 8, 0);
								pairNum++;
							}
						}

						if (static_cast<float>(pairNum) / class_templates[loopVTPNum][loopRotateNum % class_templates[loopVTPNum].size()].features.size() > similarity) {
							similarity = static_cast<float>(pairNum) / class_templates[loopVTPNum][loopRotateNum % class_templates[loopVTPNum].size()].features.size();
							i_temp = i;
							j_temp = j;
							tAngle = loopRotateNum % class_templates[loopVTPNum].size();
						}
						pairNumPre = pairNum;
					}
				}
			}
			offset_x = i_temp;
			offset_y = j_temp;
			localAngle = tAngle;
		}
		
		 std::vector<cv::Point2f> vPoints;
        for (auto k : class_templates[loopVTPNum][localAngle].features) {
            cv::circle(matchSrc, cv::Point(k.x + offset_x, k.y + offset_y), 3, cv::Scalar(255, 255, 255), -1, 8, 0);
            vPoints.emplace_back(cv::Point2f(k.x + offset_x, k.y + offset_y));
        }
		cv::RotatedRect box;
		cv::Point2f rect_point[4];

		box = cv::minAreaRect(vPoints);
		box.points(rect_point);

		for (int j = 0; j < 4; j++) {
			//line(matchSrc, rect_point[j], rect_point[(j + 1) % 4], cv::Scalar(255), 3, 8);  //绘制最小外接矩形每条边
		}
		cv::imwrite("E:/SmartMore/WorkSpace/LineModAcc/linemodacc/linemodacc/res/src" + std::to_string(nameNumb) + std::to_string(loopVTPNum) + ".png",
			matchSrc);
	}
	endTime = clock(); //计时结束
	std::cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
	if (similarity < 0.2) {
		return false;
	}
	return true;
}



bool Detector::save(std::string& dir) {
    std::ofstream fout(dir + "/config.yaml");

    YAML::Node config;

    int rotateNum = (vRotate[1] - vRotate[0]) / vRotate[2];
    int pyramidNum = T_at_level.size();

    config["PyramidNum"] = pyramidNum;
    config["rotateNum"] = rotateNum;
    config["srcImageH"] = srcImage.rows;
    config["srcImageW"] = srcImage.cols;

    YAML::Node q = YAML::Load("[]");
    config["sumVal"] = q;
    for (int cnt = 0; cnt < sumVal.size(); cnt++) {
        config["sumVal"].push_back(sumVal[cnt]);
    }
    for (int i = 0; i < pyramidNum; i++) {
        for (int j = 0; j < rotateNum; j++) {
            for (int k = 0; k < class_templates[i][j].features.size(); k++) {
                YAML::Node p = YAML::Load("[]");
                config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][std::to_string(k)] = p;
                config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][std::to_string(k)].push_back(
                    class_templates[i][j].features[k].x);
                config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][std::to_string(k)].push_back(
                    class_templates[i][j].features[k].y);
                config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][std::to_string(k)].push_back(
                    class_templates[i][j].features[k].label);
            }
        }
    }

    fout << config;
    fout.close();
    return true;
}

bool Detector::load(std::string& dir) {
    class_templates.clear();
    sumVal.clear();
    std::vector<Feature> vFeaturePoint;
    std::vector<Template> vTemplate;

    YAML::Node config = YAML::LoadFile(dir + "/config.yaml");

    int rotateNum = config["rotateNum"].as<int>();
    int PyramidNum = config["PyramidNum"].as<int>();
    srcSize.height = config["srcImageH"].as<int>();
    srcSize.width = config["srcImageW"].as<int>();
    for (int cnt = 0; cnt < config["sumVal"].size(); cnt++) {
        sumVal.emplace_back(config["sumVal"][cnt].as<int>());
    }
    for (int i = 0; i < PyramidNum; i++) {
        vTemplate.clear();
        for (int j = 0; j < rotateNum; j++) {
            vFeaturePoint.clear();
            for (int k = 0; k < config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)].size(); k++) {
                vFeaturePoint.push_back(
                    Feature(config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][k][0].as<float>(),
                            config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][k][1].as<float>(),
                            config["Pyramid" + std::to_string(i)]["Angle" + std::to_string(j)][k][2].as<int>()));
            }
            vTemplate.emplace_back(Template(vFeaturePoint));
        }
        class_templates.emplace_back(vTemplate);
    }
    if (class_templates.empty()) {
        return false;
    }
    return true;
}

// toolFunctions
bool isExitPoint(cv::Point2f p, std::vector<cv::Point2f>& vp, int index) {
    if (p.x < 0 || p.y < 0) {
        return false;
    }
    int low = 0;
    int high = vp.size() - 1;
    int mid = 0;
    bool flag = true;

    while (low <= high && flag) {
        mid = (low + high) / 2;
        if (abs(vp[mid].x - p.x) <= index) {
            flag = false;
        } else if (vp[mid].x - p.x < (-1) * index) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    if (flag) {
        return false;
    }
    for (int i = mid; i < vp.size(); i++) {
        if (abs(vp[i].x - p.x) <= index && abs(vp[i].y - p.y) <= index) {
            return true;
        } else if (vp[i].x - p.x > index) {
            break;
        }
    }

    for (int i = mid; i >= 0; i--) {
        if (abs(vp[i].x - p.x) < index && abs(vp[i].y - p.y) < index) {
            return true;
        } else if (vp[i].x - p.x < (-1) * index) {
            break;
        }
    }
    return false;
}
