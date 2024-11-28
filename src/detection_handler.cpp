#include "detection_handler.h"
#include "/root/scarlet/alexab/darknet/include/darknet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

struct DetectionResult
{
    int class_id;      // Class ID
    float prob;        // Detection probability
    int x, y;          // Top-left coordinates of bounding box
    int width, height; // Width and height of bounding box
};

class DetectionHandler
{
public:
    DetectionResult ball;
    DetectionResult ballnet;
    DetectionResult table;

    int scoreP1 = 0;
    int scoreP2 = 0;
    std::chrono::steady_clock::time_point lastScoreTime;

    DetectionHandler()
    {
        lastScoreTime = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    }

    void processFrame(cv::Mat *show_img, detection *dets, int nboxes, float thresh, char **names, int classes)
    {
        reset();

        // Process detections
        for (int i = 0; i < nboxes; ++i)
        {
            DetectionResult det = extractDetection(show_img, dets[i], thresh, names, classes);
            if (det.class_id < 0|| det.prob < 0.85)
                continue;

            if (strcmp(names[det.class_id], "ball") == 0)
            {
                ball = det;
                drawBoundingBoxWithLabel(show_img, det, "Ball", cv::Scalar(255, 50, 50));
            }
            else if (strcmp(names[det.class_id], "net") == 0)
            {
                ballnet = det;
                drawBoundingBoxWithLabel(show_img, det, "Net", cv::Scalar(50, 255, 50));
            }
            else if (strcmp(names[det.class_id], "table") == 0)
            {
                table = det;
                drawBoundingBoxWithLabel(show_img, det, "Table", cv::Scalar(50, 50, 255));
            }
        }

        // Perform scoring logic with a 1-second buffer
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastScoreTime).count() >= 1)
        {
            scoringLogic();
        }

        // Render output
        displayResults(show_img);
        displayScores();
    }

private:
    void reset()
    {
        ball = {0};
        ballnet = {0};
        table = {0};
    }
 
    DetectionResult extractDetection(cv::Mat *show_img, detection &det, float thresh, char **names, int classes)
    {
        int class_id = -1;
        float max_prob = 0;

        for (int j = 0; j < classes; ++j)
        {
            if (det.prob[j] > thresh)
            {
                max_prob = det.prob[j];
                class_id = j;
            }
        }

        if (class_id < 0)
            return {0};

        box b = det.bbox;
        int x = (b.x - b.w / 2.0) * show_img->cols;
        int y = (b.y - b.h / 2.0) * show_img->rows;
        int width = b.w * show_img->cols;
        int height = b.h * show_img->rows;

        return {class_id, max_prob, x, y, width, height};
    }

    void scoringLogic()
    {
        if (ball.width > 0 && table.width > 0)
        {
            if (ball.x < table.x)
            {
                scoreP2++;
                lastScoreTime = std::chrono::steady_clock::now();
            }
            else if (ball.x + ball.width > table.x + table.width)
            {
                scoreP1++;
                lastScoreTime = std::chrono::steady_clock::now();
            }
        }
    }

    void displayResults(cv::Mat *show_img)
    {
        std::cout << "Ball: x=" << ball.x << ", y=" << ball.y << ", w=" << ball.width << ", h=" << ball.height << std::endl;
        std::cout << "Net: x=" << ballnet.x << ", y=" << ballnet.y << ", w=" << ballnet.width << ", h=" << ballnet.height << std::endl;
        std::cout << "Table: x=" << table.x << ", y=" << table.y << ", w=" << table.width << ", h=" << table.height << std::endl;
    }

    void displayScores()
    {
        std::cout << "\nP1: " << std::setw(3) << scoreP1 << "       P2: " << std::setw(3) << scoreP2 << "\n"
                  << std::endl;
    }

    void drawBoundingBoxWithLabel(cv::Mat* show_img, const DetectionResult& det, const std::string& label, cv::Scalar color) {
        cv::Rect rect(det.x, det.y, det.width, det.height);
        cv::rectangle(*show_img, rect, color, 2);

        // Label includes confidence and coordinates
        std::string labelText = label + " " + std::to_string(static_cast<int>(det.prob * 100)) + "% (" +
                                std::to_string(det.x) + ", " + std::to_string(det.y) + ")";
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::Point textOrigin(det.x, det.y - 5);
        cv::rectangle(*show_img, textOrigin + cv::Point(0, baseline), textOrigin + cv::Point(textSize.width, -textSize.height), color, cv::FILLED);
        cv::putText(*show_img, labelText, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
};

// Global handler instance (can also be instantiated locally if needed)
DetectionHandler handler;

// Wrapper function exposed to C
extern "C" void process_frame(void *show_img, detection *dets, int nboxes, float thresh, char **names, int classes)
{
    std::cout << "Inside process_frame" << std::endl;

    if (!show_img)
    {
        std::cerr << "Error: show_img is null!" << std::endl;
        return;
    }

    if (!dets)
    {
        std::cerr << "Error: dets (detections) is null!" << std::endl;
        return;
    }

    if (!names)
    {
        std::cerr << "Error: names is null!" << std::endl;
        return;
    }

    if (nboxes <= 0)
    {
        std::cerr << "Error: nboxes is invalid: " << nboxes << std::endl;
        return;
    }

    if (classes <= 0)
    {
        std::cerr << "Error: classes is invalid: " << classes << std::endl;
        return;
    }

    handler.processFrame(static_cast<cv::Mat *>(show_img), dets, nboxes, thresh, names, classes);

    std::cout << "Exiting process_frame" << std::endl;
}
