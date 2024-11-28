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
    cv::Point prevBallPosition = {0, 0}; // Previous ball position to track movement
    std::chrono::steady_clock::time_point lastScoreTime;
    std::chrono::steady_clock::time_point lastBounceTime; // Initialize to current time

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
            if (det.class_id < 0 || det.prob < 0.85)
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

        prevBallPosition = {ball.x, ball.y};
    }

private:
    bool inBoundingBox = false; // Tracks whether the ball is in the table bounding box
    bool isBounce = false;      // Tracks if the bounce is detected
    int bounceCount = 0;        // Counter for bounces
    int bouncesP1 = 0;
    int bouncesP2 = 0;
    bool lastBounceP1 = false;                            // Tracks if the last bounce was on P1's side
    bool lastBounceP2 = false;                            // Tracks if the last bounce was on P2's side
    bool ballStartedOnP1Side = false;                     // Tracks where the ball originated
    bool ballStartedOnP2Side = false;

    bool validBounceTime = false;                    // Tracks if the ball entered the bounding box from the top
    std::chrono::steady_clock::time_point entryTime; // Time when the ball entered from the top

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

    bool detectBounce()
    {
        // Margin as a ratio of the table's width and height
        const float marginRatio = 0.3f; // 30% of the table's dimensions (adjustable)

        // Calculate margin based on the table's size
        int widthMargin = static_cast<int>(table.width * marginRatio);
        int heightMargin = static_cast<int>(table.height * marginRatio);

        // Check if ball is within the table's bounding box with tolerance
        bool insideBox = (ball.x >= table.x - widthMargin &&
                          ball.x + ball.width <= table.x + table.width + widthMargin &&
                          ball.y >= table.y - heightMargin &&
                          ball.y + ball.height <= table.y + table.height + heightMargin);

        // Determine the vertical movement direction
        bool movingDown = ball.y > prevBallPosition.y; // Ball is moving downward
        bool movingUp = ball.y < prevBallPosition.y;   // Ball is moving upward

        auto now = std::chrono::steady_clock::now();
        if (!ball.x == 0)
        {
            // Check if the ball enters from the top
            if (insideBox)
            {
                entryTime = now;        // Record the time of entry
                validBounceTime = true; // Ball entered from the top
            }

            // Reset if timeout exceeds 500ms
            if (validBounceTime &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now - entryTime).count() > 500)
            {
                validBounceTime = false; // Timeout exceeded
                inBoundingBox = false;   // Reset inBoundingBox flag
            }

            // Track if the ball remains in the bounding box
            if (validBounceTime && insideBox)
            {
                inBoundingBox = true;
            }

            if (inBoundingBox && validBounceTime)
            {
                std::cout << "In Bounding box\n";

                // Ball exits the bounding box through the top
                if (movingUp) // Exit through the top with margin
                {
                    inBoundingBox = false;
                    validBounceTime = false; // Reset flags
                    return true;             // It's a valid bounce
                }
            }
        }

        return false; // No bounce detected
    }

    void resetBounceFlags()
    {
        lastBounceP1 = false;
        lastBounceP2 = false;
        ballStartedOnP1Side = false;
        ballStartedOnP2Side = false;
    }

    void scoringLogic()
    {
        isBounce = detectBounce(); // Check for a bounce
        auto now = std::chrono::steady_clock::now();

        if (isBounce)
        {
            lastBounceTime = now; // Update the time of the last bounce
            bounceCount++;

            // Determine which side the bounce occurred on
            if (ball.x + ball.width / 2 < ballnet.x + ballnet.width / 2) // Ball on P1's side
            {
                bouncesP1++;
                if (lastBounceP1) // Consecutive bounce on P1 side
                {
                    scoreP2++;
                    std::cout << "Point for P2! Ball bounced twice on P1's side.\n";
                    resetBounceFlags();
                }
                else
                {
                    lastBounceP1 = true;
                    lastBounceP2 = false;
                    ballStartedOnP1Side = true; // Track ball's origin
                    ballStartedOnP2Side = false;
                }
            }
            else if (ball.x + ball.width / 2 > ballnet.x + ballnet.width / 2) // Ball on P2's side
            {
                bouncesP2++;
                if (lastBounceP2) // Consecutive bounce on P2 side
                {
                    scoreP1++;
                    std::cout << "Point for P1! Ball bounced twice on P2's side.\n";
                    resetBounceFlags();
                }
                else
                {
                    lastBounceP2 = true;
                    lastBounceP1 = false;
                    ballStartedOnP2Side = true; // Track ball's origin
                    ballStartedOnP1Side = false;
                }
            }
        }
        else
        {
            // Check if 3 seconds have elapsed since the last bounce
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastBounceTime).count() > 3)
            {
                if (ballStartedOnP1Side)
                {
                    scoreP2++;
                    std::cout << "Timeout! Point for P2 due to no bounce for 2 seconds.\n";
                }
                else if (ballStartedOnP2Side)
                {
                    scoreP1++;
                    std::cout << "Timeout! Point for P1 due to no bounce for 2 seconds.\n";
                }
                resetBounceFlags(); // Reset for the next rally
            }
        }
    }

    std::string getMovementDirection()
    {
        if (ball.y > prevBallPosition.y)
        {
            return "Moving Down (Previous Position: x=" + std::to_string(prevBallPosition.x) + ", y=" + std::to_string(prevBallPosition.y) + ")";
        }
        else if (ball.y < prevBallPosition.y)
        {
            return "Moving Up (Previous Position: x=" + std::to_string(prevBallPosition.x) + ", y=" + std::to_string(prevBallPosition.y) + ")";
        }
        return "Stationary (Previous Position: x=" + std::to_string(prevBallPosition.x) + ", y=" + std::to_string(prevBallPosition.y) + ")";
    }

    void displayResults(cv::Mat *show_img)
    {
        // Get the movement direction using the modular function
        std::string movementDirection = getMovementDirection();

        // Display ball, net, and table details
        std::cout << "Ball: x=" << ball.x << ", y=" << ball.y
                  << ", w=" << ball.width << ", h=" << ball.height
                  << ", Direction: " << movementDirection << std::endl;

        std::cout << "Net: x=" << ballnet.x << ", y=" << ballnet.y
                  << ", w=" << ballnet.width << ", h=" << ballnet.height << std::endl;

        std::cout << "Table: x=" << table.x << ", y=" << table.y
                  << ", w=" << table.width << ", h=" << table.height << std::endl;
    }

    void displayScores()
    {

        std::cout << "\nP1: " << std::setw(3) << scoreP1
                  << "       P2: " << std::setw(3) << scoreP2
                  << "       Total Bounces: " << std::setw(3) << bounceCount
                  << "       P1 Bounces: " << std::setw(3) << bouncesP1
                  << "       P2 Bounces: " << std::setw(3) << bouncesP2 << "\n"
                  << std::endl;
    }

    void drawBoundingBoxWithLabel(cv::Mat *show_img, const DetectionResult &det, const std::string &label, cv::Scalar color)
    {
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
