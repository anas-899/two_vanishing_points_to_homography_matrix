#include <opencv2/core.hpp>

cv::Mat homography_from_2_vanishing_points(const cv::Size& image_size,
                                           const cv::Vec3d& vp1,
                                           const cv::Vec3d& vp2,
                                           cv::Size& output_shape,
                                           bool clip = true, 
                                           size_t clip_factor = 6) {

    cv::Vec3d vanishing_line = vp1.cross(vp2);
    cv::Mat H = cv::Mat::eye(cv::Size(3, 3), CV_64F);
    H.at<double>(2, 0) = vanishing_line[0] / vanishing_line[2];
    H.at<double>(2, 1) = vanishing_line[1] / vanishing_line[2];
    H.at<double>(2, 2) = vanishing_line[2] / vanishing_line[2];
    H = H / H.at<double>(2, 2);

    cv::Mat v_post1 = H * cv::Mat(vp1);
    cv::Mat v_post2 = H * cv::Mat(vp2);
    v_post1 = v_post1 / cv::sqrt(pow(v_post1.at<double>(0, 0), 2) + pow(v_post1.at<double>(0, 1), 2));
    v_post2 = v_post2 / cv::sqrt(pow(v_post2.at<double>(0, 0),2) + pow(v_post2.at<double>(0, 1),2));

    cv::Vec4d directions_0 = { v_post1.at<double>(0, 0), -v_post1.at<double>(0, 0), v_post2.at<double>(0, 0), -v_post2.at<double>(0, 0) };
    cv::Vec4d directions_1 = { v_post1.at<double>(0, 1), -v_post1.at<double>(0, 1), v_post2.at<double>(0, 1), -v_post2.at<double>(0, 1) };
    double t_0 = std::atan2(directions_0[0], directions_1[0]);
    double t_1 = std::atan2(directions_0[1], directions_1[1]);
    double t_2 = std::atan2(directions_0[2], directions_1[2]);
    double t_3 = std::atan2(directions_0[3], directions_1[3]);
    cv::Vec4d thetas = { t_0, t_1, t_2, t_3 };
    
    double min, max;
    cv::Point min_h_ind, max_h_ind;
    cv::minMaxLoc(cv::abs(cv::Mat(thetas)), &min, &max, &min_h_ind, &max_h_ind);
    
    int h_ind = min_h_ind.y;
    bool h_ind_2 = (int)h_ind / 2 == 0;

    int v_ind = 0;
    if (h_ind_2) {
        v_ind = thetas[2] > thetas[3] ? 2 : 3;
    } 
    else {
        v_ind = thetas[2] > thetas[3] ? 0 : 1;
    }

    double A1_data[3][3] = { { directions_0[v_ind], directions_0[h_ind], 0 },
                             { directions_1[v_ind], directions_1[h_ind], 0 },
                             { 0, 0, 1 } };
    cv::Mat A1 = cv::Mat(3, 3, CV_64F, A1_data);
    if (cv::determinant(A1) < 0) {
        A1.at<double>(0, 0) = -A1.at<double>(0, 0);
        A1.at<double>(1, 0) = -A1.at<double>(1, 0);
        A1.at<double>(2, 0) = -A1.at<double>(2, 0);
    }
    
    cv::Mat A = A1.inv();
    cv::Mat inter_matrix = A * H;

    double image_data[3][4] = { { 0, 0, image_size.width, image_size.width },
                                { 0, image_size.height, 0, image_size.height },
                                { 1, 1, 1, 1 } };
    cv::Mat image_mat = cv::Mat(3, 4, CV_64F, image_data);

    cv::Mat cords = inter_matrix * image_mat;
    auto r_1 = cords.row(0) / cords.row(2);
    auto r_2 = cords.row(1) / cords.row(2);

    double min_c1, max_c1;
    cv::minMaxLoc(r_1, &min_c1, &max_c1);
    double min_c2, max_c2;
    cv::minMaxLoc(r_2, &min_c2, &max_c2);

    double tx = std::min(0.0, min_c1);
    double ty = std::min(0.0, min_c2);

    double max_x = max_c1 - tx;
    double max_y = max_c2 - ty;

    if (clip) {
        // These might be too large.Clip them.
        double max_offset = std::max(image_size.width, image_size.height) * clip_factor / 2;
        tx = std::max(tx, -max_offset);
        ty = std::max(ty, -max_offset);

        max_x = std::min(max_x, -tx + max_offset);
        max_y = std::min(max_y, -ty + max_offset);
    }

    int max_x_int = int(max_x);
    int max_y_int = int(max_y);

    double T_data[3][3] = { { 1, 0, -tx },
                            { 0, 1, -ty },
                            { 0, 0, 1 } };
    cv::Mat T = cv::Mat(3, 3, CV_64F, T_data);

    cv::Mat final_homography = T * inter_matrix;
    
    output_shape = cv::Size(max_x_int, max_y_int);
    return final_homography;
}   
