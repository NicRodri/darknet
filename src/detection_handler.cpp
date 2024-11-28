#include "detection_handler.h"
#include "/home/jeff/src/alex/darknet/include/darknet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream> // For file handling

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
    DetectionResult prevBall;
    DetectionResult ball;
    DetectionResult ballnet;
    DetectionResult table;

    int scoreP1 = 0;
    int scoreP2 = 0;
    int moving_right = -1;
    int player_to_score = 0;
    int bounce = 0;
    int bounce_num = 0;
    std::chrono::steady_clock::time_point lastScoreTime;
    std::chrono::steady_clock::time_point pass_over;

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
        directionDetection();

        
        
        
        if ((ball.y - prevBall.y) < -100 && (ball.y != 0 || prevBall.y != 0))
    {
        bounce = 1;

        // Log bounce information to a file
        std::ofstream logFile("bounce_log.txt", std::ios::app); // Open file in append mode
        if (logFile.is_open())
        {
            logFile << "Bounce #" << bounce_num << ": ball.y = " << ball.y << ", prevBall.y = " << prevBall.y << "\n";
            logFile.close();
        }
        else
        {
            printf("donefucked up\n");
            std::cerr << "Error: Unable to open file for logging!" << std::endl;
        }

        bounce_num++;
    }

        printf("moving right ? %d \n", moving_right);

        auto now = std::chrono::steady_clock::now();



        if (ball.x > ballnet.x && moving_right == 1&& player_to_score !=1){
            player_to_score = 1;
            pass_over = now;
            bounce =0;
        }

        if (ball.x < ballnet.x && moving_right == 0 && player_to_score !=2){
            player_to_score = 2;
            pass_over = now;
            bounce =0;
        }

        // if (pass_over == now){
        //     bounce =0;
        // }

        printf("bounce %d \n", bounce);
        printf("player to score %d \n", player_to_score);
        printf("prevBall y %d \n", prevBall.y);
        if (std::chrono::duration_cast<std::chrono::seconds>(now - pass_over).count() >= 1 && bounce ==1 &&std::chrono::duration_cast<std::chrono::seconds>(now - lastScoreTime).count() >= 1){
            scoringLogic();
            player_to_score = 0;
            moving_right = -1;
            bounce =0;
        }
        // if (std::chrono::duration_cast<std::chrono::seconds>(now - lastScoreTime).count() >= 1 && moving_right != -1)
        // {   
            
        //     scoringLogic(moving_right);
        // }

        // Render output
        displayResults(show_img);
        displayScores(show_img);
        prevBall = ball;
    }

private:
    

    void directionDetection(){
        if ((prevBall.x - ball.x) > 30){
                moving_right = 0;
            } else if ((prevBall.x - ball.x) < -30) {
                moving_right = 1;
            }
        if (prevBall.x == 0 || ball.x ==0|| ((prevBall.x - ball.x) < 30 && (prevBall.x - ball.x) > -30)){
            moving_right =-1;
        }
    }

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

    void scoringLogic(){
        if (player_to_score == 1){
            scoreP1++;
            bounce =0;
        } else if (player_to_score == 2){
            scoreP2++;
            bounce =0;
        }

    }

    // void scoringLogic(int move_right)
    // {
    //     if (ball.width > 0 && table.width > 0)
    //     {
    //         if ((ball.x - table.x) < -500 && move_right == 0)
    //         {
    //             scoreP2++;
    //             lastScoreTime = std::chrono::steady_clock::now();
    //         }
    //         else if (((ball.x + ball.width) - (table.x + table.width) > 500) && move_right == 1)
    //         {
    //             scoreP1++;
    //             lastScoreTime = std::chrono::steady_clock::now();
    //         }
    //     }
    // }

    void displayResults(cv::Mat *show_img)
    {
        std::cout << "Ball: x=" << ball.x << ", y=" << ball.y << ", w=" << ball.width << ", h=" << ball.height << std::endl;
        std::cout << "Net: x=" << ballnet.x << ", y=" << ballnet.y << ", w=" << ballnet.width << ", h=" << ballnet.height << std::endl;
        std::cout << "Table: x=" << table.x << ", y=" << table.y << ", w=" << table.width << ", h=" << table.height << std::endl;
    }

    void displayScores(cv::Mat *show_img)
    {
        std::cout << "\nP1: " << std::setw(3) << scoreP1 << "       P2: " << std::setw(3) << scoreP2 << "\n" << std::endl;
        char score_text[100];
        sprintf(score_text, "P1: %d | P2: %d", scoreP1, scoreP2);
        cv::putText(*show_img, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 255), 2);
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
