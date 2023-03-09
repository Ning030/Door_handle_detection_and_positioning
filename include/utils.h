#pragma once

enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};


struct Image_Point{
    float z;
    float x;
    float y;
};

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};